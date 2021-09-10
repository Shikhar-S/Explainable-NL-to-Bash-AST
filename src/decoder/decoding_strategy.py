import torch
from copy import deepcopy
import torch
import torch.nn.functional as F

from decoder.penalties import PenaltyBuilder
from model.utils import tile
from main_utils import get_logger
logger = get_logger()

class GNMTGlobalScorer(object):
    """NMT re-ranking.

    Args:
       alpha (float): Length parameter.
       beta (float):  Coverage parameter.
       length_penalty (str): Length penalty strategy.
       coverage_penalty (str): Coverage penalty strategy.

    Attributes:
        alpha (float): See above.
        beta (float): See above.
        length_penalty (callable): See :class:`penalties.PenaltyBuilder`.
        coverage_penalty (callable): See :class:`penalties.PenaltyBuilder`.
        has_cov_pen (bool): See :class:`penalties.PenaltyBuilder`.
        has_len_pen (bool): See :class:`penalties.PenaltyBuilder`.
    """

    def __init__(self, alpha=0.0, beta=0.0, length_penalty='none', coverage_penalty='none'):
        self._validate(alpha, beta, length_penalty, coverage_penalty)
        self.alpha = alpha
        self.beta = beta
        penalty_builder = PenaltyBuilder(coverage_penalty, length_penalty)
        self.has_cov_pen = penalty_builder.has_cov_pen
        # Term will be subtracted from probability
        self.cov_penalty = penalty_builder.coverage_penalty

        self.has_len_pen = penalty_builder.has_len_pen
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty

    @classmethod
    def _validate(cls, alpha, beta, length_penalty, coverage_penalty):
        # these warnings indicate that either the alpha/beta
        # forces a penalty to be a no-op, or a penalty is a no-op but
        # the alpha/beta would suggest otherwise.
        if length_penalty is None or length_penalty == "none":
            if alpha != 0:
                logger.warn("Non-default `alpha` with no length penalty. "
                              "`alpha` has no effect.")
        else:
            # using some length penalty
            if length_penalty == "wu" and alpha == 0.:
                logger.warn("Using length penalty Wu with alpha==0 "
                              "is equivalent to using length penalty none.")
        if coverage_penalty is None or coverage_penalty == "none":
            if beta != 0:
                logger.warn("Non-default `beta` with no coverage penalty. "
                              "`beta` has no effect.")
        else:
            # using some coverage penalty
            if beta == 0.:
                logger.warn("Non-default coverage penalty with beta==0 "
                              "is equivalent to using coverage penalty none.")


class DecodeStrategy(object):
    """Base class for generation strategies.

    Args:
        pad (int): Magic integer in output vocab.
        bos (int): Magic integer in output vocab.
        eos (int): Magic integer in output vocab.
        unk (int): Magic integer in output vocab.
        batch_size (int): Current batch size.
        parallel_paths (int): Decoding strategies like beam search
            use parallel paths. Each batch is repeated ``parallel_paths``
            times in relevant state tensors.
        min_length (int): Shortest acceptable generation, not counting
            begin-of-sentence or end-of-sentence.
        max_length (int): Longest acceptable sequence, not counting
            begin-of-sentence (presumably there has been no EOS
            yet if max_length is used as a cutoff).
        ban_unk_token (Boolean): Whether unk token is forbidden
        block_ngram_repeat (int): Block beams where
            ``block_ngram_repeat``-grams repeat.
        exclusion_tokens (set[int]): If a gram contains any of these
            tokens, it may repeat.
        return_attention (bool): Whether to work with attention too. If this
            is true, it is assumed that the decoder is attentional.

    Attributes:
        pad (int): See above.
        bos (int): See above.
        eos (int): See above.
        unk (int): See above.
        predictions (list[list[LongTensor]]): For each batch, holds a
            list of beam prediction sequences.
        scores (list[list[FloatTensor]]): For each batch, holds a
            list of scores.
        attention (list[list[FloatTensor or list[]]]): For each
            batch, holds a list of attention sequence tensors
            (or empty lists) having shape ``(step, inp_seq_len)`` where
            ``inp_seq_len`` is the length of the sample (not the max
            length of all inp seqs).
        alive_seq (LongTensor): Shape ``(B x parallel_paths, step)``.
            This sequence grows in the ``step`` axis on each call to
            :func:`advance()`.
        is_finished (ByteTensor or NoneType): Shape
            ``(B, parallel_paths)``. Initialized to ``None``.
        alive_attn (FloatTensor or NoneType): If tensor, shape is
            ``(step, B x parallel_paths, inp_seq_len)``, where ``inp_seq_len``
            is the (max) length of the input sequence.
        min_length (int): See above.
        max_length (int): See above.
        ban_unk_token (Boolean): See above.
        block_ngram_repeat (int): See above.
        exclusion_tokens (set[int]): See above.
        return_attention (bool): See above.
        done (bool): See above.
    """

    def __init__(self, bos, eos, eoc, unk, batch_size, parallel_paths,
                 global_scorer, min_length, block_ngram_repeat,
                 exclusion_tokens, return_attention, max_length,
                 ban_unk_token,max_level):

        # magic indices
        self.bos = bos
        self.eos = eos
        self.eoc = eoc #end of child
        self.unk = unk
        self.max_level = max_level

        self.batch_size = batch_size
        self.parallel_paths = parallel_paths
        self.global_scorer = global_scorer

        # result caching
        self.predictions = [[] for _ in range(batch_size)]
        self.scores = [[] for _ in range(batch_size)]
        self.attention = [[] for _ in range(batch_size)]
        self.hypotheses = [[] for _ in range(batch_size)]

        self.alive_attn = None

        self.min_length = min_length
        self.max_length = max_length
        self.ban_unk_token = ban_unk_token

        self.block_ngram_repeat = block_ngram_repeat
        n_paths = batch_size * parallel_paths
        self.forbidden_tokens = [dict() for _ in range(n_paths)]

        self.exclusion_tokens = exclusion_tokens
        self.return_attention = return_attention

        self.done = False

    def get_device_from_memory_bank(self, memory_bank):
        if isinstance(memory_bank, tuple):
            mb_device = memory_bank[0].device
        else:
            mb_device = memory_bank.device
        return mb_device

    def initialize_tile(self, memory_bank, guidance, src_lengths, src_map=None):
        def fn_map_state(state, dim):
            return tile(state, self.beam_size, dim=dim)

        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, self.beam_size, dim=1) for x in memory_bank)
        elif memory_bank is not None:
            memory_bank = tile(memory_bank, self.beam_size, dim=1)
        if src_map is not None:
            src_map = tile(src_map, self.beam_size, dim=1)

        self.memory_lengths = tile(src_lengths, self.beam_size, dim=0)
        guidance = tile(guidance,self.beam_size, dim = 0)

        return fn_map_state, memory_bank, guidance, src_map

    def initialize(self, memory_bank, src_lengths, src_map=None, device=None):
        """DecodeStrategy subclasses should override :func:`initialize()`.

        `initialize` should be called before all actions.
        used to prepare necessary ingredients for decode.
        """
        if device is None:
            device = torch.device('cpu')
        self.alive_seq = torch.full(
            [self.batch_size * self.parallel_paths, 1], self.bos,
            dtype=torch.long, device=device)
        self.alive_parent_idx = torch.zeros([self.batch_size * self.parallel_paths,1], dtype = torch.long, device=device)
        self.tree_coordinates = torch.zeros([self.batch_size * self.parallel_paths,1], dtype = torch.long, device=device)
        self.is_finished = torch.zeros([self.batch_size, self.parallel_paths], dtype=torch.uint8, device=device)
        return None, memory_bank, src_lengths, src_map

    def __len__(self):
        return self.alive_seq.shape[1]

    # In below ensure* functions negative infinity in score makes the particular path unlikely.
    def ensure_min_length(self, log_probs):
        if len(self) <= self.min_length and self.is_type_token:
            log_probs[:, self.eos] = -1e20

    def ensure_unk_removed(self, log_probs):
        if self.ban_unk_token:
            log_probs[:, self.unk] = -1e20

    def ensure_max_length(self):
        # add one to account for BOS. Don't account for EOS because hitting
        # this implies it hasn't been found.
        # print(len(self),self.max_length)
        if len(self) >= self.max_length + 1 and self.is_type_token:
            # print(self.max_length,len(self),'Finishing beams since they excede max length')
            self.is_finished.fill_(1)

    def ensure_eos_follows_last_ec(self):
        idx = self.alive_seq[torch.arange(self.alive_seq.shape[0]),self.alive_parent_idx.squeeze(-1)].eq(self.eoc).view(-1,self.parallel_paths)
        self.is_finished.masked_fill_(idx,1)
        
    def block_ngram_repeats(self, log_probs):
        """
        We prevent the beam from going in any direction that would repeat any
        ngram of size <block_ngram_repeat> more than once.

        The way we do it: we maintain a list of all ngrams of size
        <block_ngram_repeat> that is updated each time the beam advances, and
        manually put any token that would lead to a repeated ngram to 0.

        This improves on the previous version's complexity:
           - previous version's complexity: batch_size * beam_size * len(self)
           - current version's complexity: batch_size * beam_size

        This improves on the previous version's accuracy;
           - Previous version blocks the whole beam, whereas here we only
            block specific tokens.
           - Before the translation would fail when all beams contained
            repeated ngrams. This is sure to never happen here.
        """

        # we don't block anything if the user doesn't want it
        if self.block_ngram_repeat <= 0:
            return

        # we can't block anything: beam's too short
        if len(self) < self.block_ngram_repeat:
            return

        n = self.block_ngram_repeat - 1
        for path_idx in range(self.alive_seq.shape[0]):
            # we check paths one by one

            current_ngram = tuple(self.alive_seq[path_idx, -n:].tolist())
            forbidden_tokens = self.forbidden_tokens[path_idx].get(
                current_ngram, None)
            if forbidden_tokens is not None:
                log_probs[path_idx, list(forbidden_tokens)] = -10e20

    def maybe_update_forbidden_tokens(self):
        """We complete and reorder the list of forbidden_tokens"""

        # we don't forbid anything if the user doesn't want it
        if self.block_ngram_repeat <= 0:
            return

        # we can't forbid anything if beam's too short
        if len(self) < self.block_ngram_repeat:
            return

        n = self.block_ngram_repeat

        forbidden_tokens = list()
        for path_idx, seq in zip(self.select_indices, self.alive_seq):

            # Reordering forbidden_tokens following beam selection
            # We rebuild a dict to ensure we get the value and not the pointer
            forbidden_tokens.append(
                deepcopy(self.forbidden_tokens[path_idx]))

            # Grabbing the newly selected tokens and associated ngram
            current_ngram = tuple(seq[-n:].tolist())

            # skip the blocking if any token in current_ngram is excluded
            if set(current_ngram) & self.exclusion_tokens:
                continue

            forbidden_tokens[-1].setdefault(current_ngram[:-1], set())
            forbidden_tokens[-1][current_ngram[:-1]].add(current_ngram[-1])

        self.forbidden_tokens = forbidden_tokens

    def advance(self, log_probs, attn, is_type_token):
        """DecodeStrategy subclasses should override :func:`advance()`.

        Advance is used to update ``self.alive_seq``, ``self.is_finished``,
        and, when appropriate, ``self.alive_attn``.
        """

        raise NotImplementedError()
    
    def increment_parent(self,topk_ids):
        prev_increment_parents = torch.zeros([topk_ids.shape[0],1]).type_as(topk_ids)
        new_increment_parents = 2 * topk_ids.eq(self.eoc).long()
        any_parent_is_ec = True
        remaining_space = torch.full((self.alive_seq.shape[0],1), fill_value = self.alive_seq.shape[1]-1,dtype=torch.long)\
                                        .type_as(new_increment_parents) - self.alive_parent_idx
        while(any_parent_is_ec):
            
            new_increment_parents = torch.min(new_increment_parents,remaining_space)
            assert (self.alive_parent_idx + new_increment_parents < self.alive_seq.shape[1]).all()

            new_parent_idx = (self.alive_parent_idx + new_increment_parents).squeeze(-1)
            any_parent_is_ec = self.alive_seq[torch.arange(self.alive_seq.shape[0]).type_as(self.alive_seq),\
                                                new_parent_idx].eq(self.eoc).any()
            if new_increment_parents.eq(prev_increment_parents).all(): #no change
                break
            if not any_parent_is_ec:
                break
            prev_increment_parents = new_increment_parents.clone()
            new_increment_parents += 2 * self.alive_seq[torch.arange(self.alive_seq.shape[0]).type_as(self.alive_seq),\
                                                        new_parent_idx].eq(self.eoc).long().unsqueeze(-1)
        # print('Incrementing with--',new_increment_parents.squeeze(1)[:10],new_increment_parents.shape)
        # print('Alive parent index--',self.alive_parent_idx.squeeze(1)[:10],self.alive_parent_idx.shape)
        self.alive_parent_idx = self.alive_parent_idx + new_increment_parents
        # print('After incrementing',self.alive_parent_idx.squeeze(1)[:10],self.alive_parent_idx.shape)

    def update_finished(self):
        """DecodeStrategy subclasses should override :func:`update_finished()`.

        ``update_finished`` is used to update ``self.predictions``,
        ``self.scores``, and other "output" attributes.
        """

        raise NotImplementedError()


def sample_topp(logits, keep_topp):
    sorted_logits, sorted_indices = torch.sort(logits,
                                               descending=True,
                                               dim=1)

    cumulative_probs = torch.cumsum(F.softmax(sorted_logits,
                                              dim=-1), dim=-1)
    sorted_indices_to_keep = cumulative_probs.lt(keep_topp)

    # keep indices until overflowing p
    cumsum_mask = sorted_indices_to_keep.cumsum(dim=1)
    last_included = cumsum_mask[:, -1:]
    last_included.clamp_(0, sorted_indices_to_keep.size()[1] - 1)
    sorted_indices_to_keep = sorted_indices_to_keep.scatter_(
        1, last_included, 1)

    # Set all logits that are not in the top-p to -10000.
    # This puts the probabilities close to 0.
    keep_indices = sorted_indices_to_keep.scatter(
                                1,
                                sorted_indices,
                                sorted_indices_to_keep,
                                )
    return logits.masked_fill(~keep_indices, -10000)


def sample_topk(logits, keep_topk):
    top_values, _ = torch.topk(logits, keep_topk, dim=1)
    kth_best = top_values[:, -1].view([-1, 1])
    kth_best = kth_best.repeat([1, logits.shape[1]]).float()

    # Set all logits that are not in the top-k to -10000.
    # This puts the probabilities close to 0.
    ignore = torch.lt(logits, kth_best)
    return logits.masked_fill(ignore, -10000)


def sample_with_temperature(logits, sampling_temp, keep_topk, keep_topp):
    """Select next tokens randomly from the top k possible next tokens.

    Samples from a categorical distribution over the ``keep_topk`` words using
    the category probabilities ``logits / sampling_temp``.

    Args:
        logits (FloatTensor): Shaped ``(batch_size, vocab_size)``.
            These can be logits (``(-inf, inf)``) or log-probs (``(-inf, 0]``).
            (The distribution actually uses the log-probabilities
            ``logits - logits.logsumexp(-1)``, which equals the logits if
            they are log-probabilities summing to 1.)
        sampling_temp (float): Used to scale down logits. The higher the
            value, the more likely it is that a non-max word will be
            sampled.
        keep_topk (int): This many words could potentially be chosen. The
            other logits are set to have probability 0.
        keep_topp (float): Keep most likely words until the cumulated
            probability is greater than p. If used with keep_topk: both
            conditions will be applied

    Returns:
        (LongTensor, FloatTensor):

        * topk_ids: Shaped ``(batch_size, 1)``. These are
          the sampled word indices in the output vocab.
        * topk_scores: Shaped ``(batch_size, 1)``. These
          are essentially ``(logits / sampling_temp)[topk_ids]``.
    """
    
    if sampling_temp == 0.0 or keep_topk == 1:
        # For temp=0.0, take the argmax to avoid divide-by-zero errors.
        # keep_topk=1 is also equivalent to argmax.
        topk_scores, topk_ids = logits.topk(1, dim=-1)
        if sampling_temp > 0:
            topk_scores /= sampling_temp
    else:
        logits = torch.div(logits, sampling_temp)
        if keep_topp > 0:
            logits = sample_topp(logits, keep_topp)
        if keep_topk > 0:
            logits = sample_topk(logits, keep_topk)

        dist = torch.distributions.Multinomial(logits=logits, total_count=1)
        topk_ids = torch.argmax(dist.sample(), dim=1, keepdim=True)
        topk_scores = logits.gather(dim=1, index=topk_ids)
    return topk_ids, topk_scores


class GreedySearch(DecodeStrategy):
    """Select next tokens randomly from the top k possible next tokens.

    The ``scores`` attribute's lists are the score, after applying temperature,
    of the final prediction (either EOS or the final token in the event
    that ``max_length`` is reached)

    Args:
        pad (int): See base.
        bos (int): See base.
        eos (int): See base.
        unk (int): See base.
        batch_size (int): See base.
        global_scorer (onmt.translate.GNMTGlobalScorer): Scorer instance.
        min_length (int): See base.
        max_length (int): See base.
        ban_unk_token (Boolean): See base.
        block_ngram_repeat (int): See base.
        exclusion_tokens (set[int]): See base.
        return_attention (bool): See base.
        max_length (int): See base.
        sampling_temp (float): See
            :func:`~onmt.translate.greedy_search.sample_with_temperature()`.
        keep_topk (int): See
            :func:`~onmt.translate.greedy_search.sample_with_temperature()`.
        keep_topp (float): See
            :func:`~onmt.translate.greedy_search.sample_with_temperature()`.
        beam_size (int): Number of beams to use.
    """

    def __init__(self, bos, eos, eoc, unk, batch_size, global_scorer,
                 min_length, block_ngram_repeat, exclusion_tokens,
                 return_attention, max_length, sampling_temp, keep_topk,
                 keep_topp, beam_size, ban_unk_token,max_level):
        super(GreedySearch, self).__init__(bos, eos, eoc, unk, batch_size, beam_size, global_scorer,
            min_length, block_ngram_repeat, exclusion_tokens,
            return_attention, max_length, ban_unk_token,max_level)
        self.sampling_temp = sampling_temp
        self.keep_topk = keep_topk
        self.keep_topp = keep_topp
        self.topk_scores = None
        self.beam_size = beam_size

    def initialize(self, memory_bank, src_lengths, src_map=None):
        """Initialize for decoding."""
        (fn_map_state, memory_bank,src_map) = self.initialize_tile(memory_bank, src_lengths, src_map)
        device = self.get_device_from_memory_bank(memory_bank)

        super(GreedySearch, self).initialize(
            memory_bank, src_lengths, src_map, device)
        self.select_indices = torch.arange(self.batch_size*self.beam_size, dtype=torch.long, device=device)
        self.original_batch_idx = fn_map_state(torch.arange(self.batch_size, dtype=torch.long, device=device), dim=0)
        self.beams_scores = torch.zeros((self.batch_size*self.beam_size, 1),dtype=torch.float, device=device)
        return fn_map_state, memory_bank, self.memory_lengths, src_map

    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]
    
    @property
    def current_parents(self):
        return self.alive_parent_idx[:,-1]

    @property
    def batch_offset(self):
        return self.select_indices
    
    @property
    def current_tree_coordinates(self):
        capTensor = torch.Tensor([self.max_level]).type_as(self.tree_coordinates).unsqueeze(1)
        return torch.min(self.tree_coordinates[:,-1],capTensor)

    def _pick(self, log_probs):
        """Function used to pick next tokens.

        Args:
            log_probs (FloatTensor): ``(batch_size, vocab_size)``.
        """
        topk_ids, topk_scores = sample_with_temperature(log_probs, self.sampling_temp, self.keep_topk, self.keep_topp)
        return topk_ids, topk_scores

    def align_select_indices(self):
        nb_finished_beams = (self.is_finished.view(-1).size(0) - self.select_indices.size(0))
        if nb_finished_beams:
            self.select_indices = torch.arange(
                self.select_indices.size(0), dtype=torch.long,
                device=self.select_indices.device)

    def advance(self, log_probs, attn, is_type_token):
        """Select next tokens randomly from the top k possible next tokens.

        Args:
            log_probs (FloatTensor): Shaped ``(batch_size, vocab_size)``.
                These can be logits (``(-inf, inf)``) or log-probs
                (``(-inf, 0]``). (The distribution actually uses the
                log-probabilities ``logits - logits.logsumexp(-1)``,
                which equals the logits if they are log-probabilities summing
                to 1.)
            attn (FloatTensor): Shaped ``(1, B, inp_seq_len)``.
        """
        self.is_type_token = is_type_token #Initialise step
        self.align_select_indices()

        # self.ensure_min_length(log_probs)
        # self.ensure_unk_removed(log_probs)
        # self.block_ngram_repeats(log_probs)
        if self.is_type_token:
            self.ensure_eos_follows_last_ec(log_probs)

        topk_ids, self.topk_scores = self._pick(log_probs)
        self.beams_scores += self.topk_scores

        if self.is_type_token:
            self.is_finished = topk_ids.eq(self.eos)
        else:
            self.is_finished = torch.zeros([self.batch_size,self.parallel_paths],dtype=torch.uint8,device=topk_ids.device)
        
        self.alive_seq = torch.cat([self.alive_seq, topk_ids], -1)
        
        filtered_tree_coordinates = self.tree_coordinates.clone() #artefact beam search
        idx1 = torch.arange(filtered_tree_coordinates.shape[0])
        idx2 = self.alive_parent_idx[:,-1]
        next_tree_coordinates = filtered_tree_coordinates[idx1,idx2] + 1
        self.tree_coordinates = torch.cat([filtered_tree_coordinates,next_tree_coordinates.unsqueeze(1)], dim=-1)

        if self.is_type_token:
            self.increment_parent(topk_ids)

        if self.return_attention:
            if self.alive_attn is None:
                self.alive_attn = attn
            else:
                self.alive_attn = torch.cat([self.alive_attn, attn], 0)
        self.ensure_max_length()

    def update_finished(self):
        """Finalize scores and predictions."""
        # shape: (sum(~ self.is_finished), 1)
        finished_batches = self.is_finished.view(-1).nonzero(as_tuple=False)
        step = len(self)
        length_penalty = self.global_scorer.length_penalty(step, alpha=self.global_scorer.alpha)

        for b in finished_batches.view(-1):
            b_orig = self.original_batch_idx[b]
            score = self.beams_scores[b, 0]/length_penalty
            pred = self.alive_seq[b, 1:]
            attention = (
                self.alive_attn[:, b, :self.memory_lengths[b]]
                if self.alive_attn is not None else [])
            self.hypotheses[b_orig].append((score, pred, attention))

        self.done = self.is_finished.all()
        if self.done:
            for b in range(self.batch_size):
                best_hyp = sorted(
                    self.hypotheses[b], key=lambda x: x[0], reverse=True)
                for score, pred, attn in best_hyp:
                    self.scores[b].append(score)
                    self.predictions[b].append(pred)
                    self.attention[b].append(attn)
            return
        is_alive = ~self.is_finished.view(-1)
        self.alive_parent_idx = self.alive_parent_idx[is_alive]
        self.alive_seq = self.alive_seq[is_alive]
        self.tree_coordinates = self.tree_coordinates[is_alive]
        self.beams_scores = self.beams_scores[is_alive]
        if self.alive_attn is not None:
            self.alive_attn = self.alive_attn[:, is_alive]
        self.select_indices = is_alive.nonzero(as_tuple=False).view(-1)
        self.original_batch_idx = self.original_batch_idx[is_alive]


class BeamSearchBase(DecodeStrategy):
    """Generation beam search.

    Note that the attributes list is not exhaustive. Rather, it highlights
    tensors to document their shape. (Since the state variables' "batch"
    size decreases as beams finish, we denote this axis with a B rather than
    ``batch_size``).

    Args:
        beam_size (int): Number of beams to use (see base ``parallel_paths``).
        batch_size (int): See base.
        pad (int): See base.
        bos (int): See base.
        eos (int): See base.
        unk (int): See base.
        n_best (int): Don't stop until at least this many beams have
            reached EOS.
        global_scorer (onmt.translate.GNMTGlobalScorer): Scorer instance.
        min_length (int): See base.
        max_length (int): See base.
        return_attention (bool): See base.
        block_ngram_repeat (int): See base.
        exclusion_tokens (set[int]): See base.

    Attributes:
        top_beam_finished (ByteTensor): Shape ``(B,)``.
        _batch_offset (LongTensor): Shape ``(B,)``.
        _beam_offset (LongTensor): Shape ``(batch_size x beam_size,)``.
        alive_seq (LongTensor): See base.
        topk_log_probs (FloatTensor): Shape ``(B x beam_size,)``. These
            are the scores used for the topk operation.
        memory_lengths (LongTensor): Lengths of encodings. Used for
            masking attentions.
        select_indices (LongTensor or NoneType): Shape
            ``(B x beam_size,)``. This is just a flat view of the
            ``_batch_index``.
        topk_scores (FloatTensor): Shape
            ``(B, beam_size)``. These are the
            scores a sequence will receive if it finishes.
        topk_ids (LongTensor): Shape ``(B, beam_size)``. These are the
            word indices of the topk predictions.
        _batch_index (LongTensor): Shape ``(B, beam_size)``.
        _prev_penalty (FloatTensor or NoneType): Shape
            ``(B, beam_size)``. Initialized to ``None``.
        _coverage (FloatTensor or NoneType): Shape
            ``(1, B x beam_size, inp_seq_len)``.
        hypotheses (list[list[Tuple[Tensor]]]): Contains a tuple
            of score (float), sequence (long), and attention (float or None).
    """
    def __init__(self, beam_size, batch_size, bos, eos, eoc, unk, n_best,
                 global_scorer, min_length, max_length, return_attention,
                 block_ngram_repeat, exclusion_tokens, stepwise_penalty,
                 ratio, ban_unk_token, max_level):
        super(BeamSearchBase, self).__init__(
            bos, eos, eoc, unk, batch_size, beam_size, global_scorer,
            min_length, block_ngram_repeat, exclusion_tokens,
            return_attention, max_length, ban_unk_token, max_level)
        # beam parameters
        self.beam_size = beam_size
        self.n_best = n_best
        self.ratio = ratio

        # beam state
        self.top_beam_finished = torch.zeros([batch_size], dtype=torch.uint8)
        # BoolTensor was introduced in pytorch 1.2
        try:
            self.top_beam_finished = self.top_beam_finished.bool()
        except AttributeError:
            pass
        self._batch_offset = torch.arange(batch_size, dtype=torch.long)

        self.select_indices = None
        self.done = False
        # "global state" of the old beam
        self._prev_penalty = None
        self._coverage = None

        self._stepwise_cov_pen = (
            stepwise_penalty and self.global_scorer.has_cov_pen)
        self._vanilla_cov_pen = (
            not stepwise_penalty and self.global_scorer.has_cov_pen)
        self._cov_pen = self.global_scorer.has_cov_pen

        self.memory_lengths = None

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    def initialize_(self, memory_bank, memory_lengths, src_map, device):
        super(BeamSearchBase, self).initialize(
            memory_bank, memory_lengths, src_map, device)

        self.best_scores = torch.full(
            [self.batch_size], -1e10, dtype=torch.float, device=device)
        self._beam_offset = torch.arange(
            0, self.batch_size * self.beam_size, step=self.beam_size,
            dtype=torch.long, device=device)
        self.topk_log_probs = torch.tensor(
            [0.0] + [float("-inf")] * (self.beam_size - 1), device=device
        ).repeat(self.batch_size)
        # buffers for the topk scores and 'backpointer'
        self.topk_scores = torch.empty((self.batch_size, self.beam_size),
                                       dtype=torch.float, device=device)
        self.topk_ids = torch.empty((self.batch_size, self.beam_size),
                                    dtype=torch.long, device=device)
        self._batch_index = torch.empty([self.batch_size, self.beam_size],
                                        dtype=torch.long, device=device)

    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    @property
    def current_parents(self):
        return self.alive_parent_idx[:,-1]

    @property
    def current_tree_coordinates(self):
        capTensor = torch.Tensor([self.max_level]).type_as(self.tree_coordinates).unsqueeze(1)
        return torch.min(self.tree_coordinates[:,-1],capTensor)

    @property
    def current_backptr(self):
        # for testing
        return self.select_indices.view(self.batch_size, self.beam_size)\
            .fmod(self.beam_size)

    @property
    def batch_offset(self):
        return self._batch_offset

    def _pick(self, log_probs):
        """Return token decision for a step.

        Args:
            log_probs (FloatTensor): (B, vocab_size)

        Returns:
            topk_scores (FloatTensor): (B, beam_size)
            topk_ids (LongTensor): (B, beam_size)
        """
        vocab_size = log_probs.size(-1)

        # Flatten probs into a list of possibilities.
        curr_scores = log_probs.reshape(-1, self.beam_size * vocab_size)
        topk_scores, topk_ids = torch.topk(curr_scores, self.beam_size, dim=-1) #batch X beam sizes
        return topk_scores, topk_ids

    def update_finished(self):
        # Penalize beams that finished.
        _B_old = self.topk_log_probs.shape[0]
        step = self.alive_seq.shape[-1]  # 1 greater than the step in advance
        self.topk_log_probs.masked_fill_(self.is_finished, -1e10) #This prevents finished beams from affecting other beams
        # on real data (newstest2017) with the pretrained transformer,
        # it's faster to not move this back to the original device
        self.is_finished = self.is_finished.to('cpu')
        self.top_beam_finished |= self.is_finished[:, 0].eq(1)
        predictions = self.alive_seq.view(_B_old, self.beam_size, step)
        attention = (
            self.alive_attn.view(
                step - 1, _B_old, self.beam_size, self.alive_attn.size(-1))
            if self.alive_attn is not None else None)
        non_finished_batch = []
        for i in range(self.is_finished.size(0)):  # Batch level
            b = self._batch_offset[i]
            finished_hyp = self.is_finished[i].nonzero(as_tuple=False).view(-1)
            # Store finished hypotheses for this batch.
            for j in finished_hyp:  # Beam level: finished beam j in batch i
                if self.ratio > 0:
                    s = self.topk_scores[i, j] / (step + 1)
                    if self.best_scores[b] < s:
                        self.best_scores[b] = s
                self.hypotheses[b].append((
                    self.topk_scores[i, j],
                    predictions[i, j, 1:],  # Ignore start_token.
                    attention[:, i, j, :self.memory_lengths[i]]
                    if attention is not None else None))
               
            # End condition is the top beam finished AND we can return
            # n_best hypotheses.
            if self.ratio > 0:
                pred_len = self.memory_lengths[i] * self.ratio
                finish_flag = ((self.topk_scores[i, 0] / pred_len)
                               <= self.best_scores[b]) or \
                    self.is_finished[i].all()
            else:
                finish_flag = self.top_beam_finished[i] != 0
            if finish_flag and len(self.hypotheses[b]) >= self.n_best:
                best_hyp = sorted(
                    self.hypotheses[b], key=lambda x: x[0], reverse=True)
                for n, (score, pred, attn) in enumerate(best_hyp):
                    if n >= self.n_best:
                        break
                    self.scores[b].append(score)
                    self.predictions[b].append(pred)  # ``(batch, n_best,)``
                    self.attention[b].append(
                        attn if attn is not None else [])
            else:
                non_finished_batch.append(i)
        non_finished = torch.tensor(non_finished_batch)
        # If all sentences are translated, no need to go further.
        if len(non_finished) == 0:
            self.done = True
            return

        _B_new = non_finished.shape[0]
        self.remove_finished_batches(_B_new, _B_old, non_finished,
                                     predictions, attention, step)

    def remove_finished_batches(self, _B_new, _B_old, non_finished,
                                predictions, attention, step):
        # Remove finished batches for the next step.
        self.top_beam_finished = self.top_beam_finished.index_select(0, non_finished)
        self._batch_offset = self._batch_offset.index_select(0, non_finished)
        non_finished = non_finished.to(self.topk_ids.device)
        self.topk_log_probs = self.topk_log_probs.index_select(0, non_finished)
        self._batch_index = self._batch_index.index_select(0, non_finished)
        self.select_indices = self._batch_index.view(_B_new * self.beam_size)
        self.alive_parent_idx = self.alive_parent_idx.view(predictions.shape[0],self.beam_size).index_select(0,non_finished).view(-1).unsqueeze(1)
        self.tree_coordinates = self.tree_coordinates.view(predictions.shape[0],self.beam_size,-1).index_select(0,non_finished)
        self.tree_coordinates = self.tree_coordinates.view(-1, self.tree_coordinates.size(-1))
        # self.alive_parent_idx = self.alive_parent_idx.index_select(0,self.select_indices)
        # self.tree_coordinates = self.tree_coordinates.index_select(0,self.select_indices)
        self.alive_seq = predictions.index_select(0, non_finished) \
            .view(-1, self.alive_seq.size(-1))
        self.topk_scores = self.topk_scores.index_select(0, non_finished)
        self.topk_ids = self.topk_ids.index_select(0, non_finished)
        if self.alive_attn is not None:
            inp_seq_len = self.alive_attn.size(-1)
            self.alive_attn = attention.index_select(1, non_finished) \
                .view(step - 1, _B_new * self.beam_size, inp_seq_len)
            if self._cov_pen:
                self._coverage = self._coverage \
                    .view(1, _B_old, self.beam_size, inp_seq_len) \
                    .index_select(1, non_finished) \
                    .view(1, _B_new * self.beam_size, inp_seq_len)
                if self._stepwise_cov_pen:
                    self._prev_penalty = self._prev_penalty.index_select(
                        0, non_finished)
        
    
    def advance(self,log_probs, attn, is_type_token):
        self.is_type_token = is_type_token
        vocab_size = log_probs.size(-1)

        # using integer division to get an integer _B without casting
        _B = log_probs.shape[0] // self.beam_size

        # if self._stepwise_cov_pen and self._prev_penalty is not None:
        #     self.topk_log_probs += self._prev_penalty
        #     self.topk_log_probs -= self.global_scorer.cov_penalty(
        #         self._coverage + attn, self.global_scorer.beta).view(
        #         _B, self.beam_size)

        # force the output to be longer than self.min_length
        step = len(self)
        self.ensure_min_length(log_probs)
        # self.ensure_unk_removed(log_probs)
        # if self.is_type_token:
        #     self.ensure_eos_follows_last_ec(log_probs)


        # Multiply probs by the beam probability.
        log_probs += self.topk_log_probs.view(_B * self.beam_size, 1)

        # if the sequence ends now, then the penalty is the current
        # length + 1, to include the EOS token
        length_penalty = self.global_scorer.length_penalty(step + 1, alpha=self.global_scorer.alpha)
        # assert length_penalty<=1
        #print(length_penalty)
        curr_scores = log_probs / length_penalty

        # Avoid any direction that would repeat unwanted ngrams
        # self.block_ngram_repeats(curr_scores) TODO: Check This

        # Pick up candidate token by curr_scores
        self.topk_scores, self.topk_ids = self._pick(curr_scores) #_B X beam size

        # Recover log probs.
        # Length penalty is just a scalar. It doesn't matter if it's applied
        # before or after the topk.
        self.topk_log_probs = torch.mul(self.topk_scores, length_penalty)

        # Resolve beam origin and map to batch index flat representation.
        # print('Top id--',self.topk_ids[0].unsqueeze(0))
        self._batch_index = self.topk_ids // vocab_size #now stores which beam gave the top scoring id
        self._batch_index += self._beam_offset[:_B].unsqueeze(1) #add that to offset for batch start 
        self.select_indices = self._batch_index.view(_B * self.beam_size) #reshape into a single flattened tensor
        self.topk_ids.fmod_(vocab_size)  # resolve true word ids
        # if self.is_type_token:
        #     print('-'*10)
        #     print(self.topk_ids[0])
        #     print('-'*10)
        #     print(self.alive_parent_idx[:10,0],'<-parent index')
        #     print('-'*40)
        
        # Append last prediction.
        # also stops low scoring sequences by not selecting them in index_select
        self.alive_seq = torch.cat(
            [self.alive_seq.index_select(0, self.select_indices),
             self.topk_ids.view(_B * self.beam_size, 1)], -1)

        
        
        # self.maybe_update_forbidden_tokens()

        # if self.return_attention or self._cov_pen:
        #     current_attn = attn.index_select(1, self.select_indices)
        #     if step == 1:
        #         self.alive_attn = current_attn
        #         # update global state (step == 1)
        #         if self._cov_pen:  # coverage penalty
        #             self._prev_penalty = torch.zeros_like(self.topk_log_probs)
        #             self._coverage = current_attn
        #     else:
        #         self.alive_attn = self.alive_attn.index_select(
        #             1, self.select_indices)
        #         self.alive_attn = torch.cat([self.alive_attn, current_attn], 0)
        #         # update global state (step > 1)
        #         if self._cov_pen:
        #             self._coverage = self._coverage.index_select(
        #                 1, self.select_indices)
        #             self._coverage += current_attn
        #             self._prev_penalty = self.global_scorer.cov_penalty(
        #                 self._coverage, beta=self.global_scorer.beta).view(
        #                     _B, self.beam_size)

        # if self._vanilla_cov_pen:
        #     # shape: (batch_size x beam_size, 1)
        #     cov_penalty = self.global_scorer.cov_penalty(
        #         self._coverage,
        #         beta=self.global_scorer.beta)
        #     self.topk_scores -= cov_penalty.view(_B, self.beam_size).float()
        if self.is_type_token:
            self.is_finished = self.topk_ids.eq(self.eos)
        else:
            self.is_finished = torch.zeros([1,1],dtype=torch.uint8,device=self.topk_ids.device)

        self.tree_coordinates = self.tree_coordinates.index_select(0,self.select_indices)
        self.alive_parent_idx = self.alive_parent_idx.index_select(0,self.select_indices)
        
        if self.is_type_token:
            self.increment_parent(self.alive_seq[:,-3].unsqueeze(-1))
            #add new level
            idx1 = torch.arange(self.tree_coordinates.shape[0])
            idx2 = self.alive_parent_idx[:,-1] // 2
            max_parent_idx = torch.LongTensor([self.tree_coordinates.shape[1]-1]).type_as(idx2)
            idx2 = torch.min(idx2,max_parent_idx)
            next_tree_coordinates = self.tree_coordinates[idx1,idx2] + 1
            self.tree_coordinates = torch.cat([self.tree_coordinates,next_tree_coordinates.unsqueeze(1)], dim=-1)
            self.ensure_eos_follows_last_ec()

        # print(self.alive_parent_idx)
        # print('-'*100)
        # print(self.alive_seq)
        # print('='*100)
        self.ensure_max_length()


class BeamSearch(BeamSearchBase):
    """
        Beam search for seq2seq/encoder-decoder models
    """
    def initialize(self, memory_bank, guidance, src_lengths, src_map=None, device=None):
        """Initialize for decoding.
        Repeat src objects `beam_size` times.
        """

        (fn_map_state, memory_bank, guidance, src_map) = self.initialize_tile(memory_bank, guidance, src_lengths, src_map)
        device = self.get_device_from_memory_bank(memory_bank)
        super(BeamSearch, self).initialize_(memory_bank, self.memory_lengths, src_map, device)
        return fn_map_state, memory_bank, guidance, self.memory_lengths, src_map
