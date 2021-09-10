from bashlint import nast
from bashlint import data_tools
from utils.metric_utils import compute_metric
import numpy as np

def get_node(parent,typ,val):
    if typ == 'utility':
        ret_node = nast.UtilityNode(value = val, parent = parent)
    elif typ == 'flag':
        ret_node = nast.FlagNode(value = val, parent = parent)
    elif typ == 'argument':
        ret_node = nast.ArgumentNode(value= val, arg_type= val, parent = parent)
    elif  typ == 'operator':
        ret_node = nast.OperatorNode(value = val, parent = parent)
    elif typ == 'unarylogicop':
        ret_node = nast.UnaryLogicOpNode(value = val, parent = parent)
    elif typ == 'binarylogicop':
        ret_node = nast.BinaryLogicOpNode(value = val, parent = parent)
    elif typ == 'processsubstitution':
        ret_node = nast.ProcessSubstitutionNode(value = val, parent = parent)
    elif typ == 'bracket':
        ret_node = nast.BracketNode(parent = parent)
    elif typ == 'commandsubstitution':
        ret_node = nast.CommandSubstitutionNode(parent = parent)
    elif typ == 'pipeline':
        ret_node = nast.PipelineNode(parent = parent)
    else:
        ret_node = nast.Node(parent = parent,kind = typ,value = val)
    parent.add_child(ret_node)
    return ret_node

def generate_ast(structure,value):
    queue_ptr = 1
    root = nast.Node(kind = structure[0],value = value[0])
    nodelist= [root]
    parent_ptr = 0
    while queue_ptr<len(structure):
        typ = structure[queue_ptr]
        val = value[queue_ptr]
        if typ=='ET':
            break
        if typ == 'EC':
            parent_ptr += 1
            queue_ptr += 1
            continue
        node = get_node(nodelist[parent_ptr],typ,val)
        nodelist.append(node)
        queue_ptr += 1
    return nodelist[0]


def get_score(truth_s_list,truth_v_list,pred_s_list,pred_v_list,inv=None,inv_tag=None):
    scores = []
    all_scores = []
    c=0
    batch_size = len(truth_v_list)
    assert ( len(truth_s_list) == len(truth_v_list) and len(truth_v_list) == len(pred_v_list))
    assert len(truth_s_list) == len(pred_s_list)
    for b in range(batch_size):
        truth = data_tools.ast2command(generate_ast(truth_s_list[b][0],truth_v_list[b][0]))
        preds = []
        pred_indices = []
        pred_score = [-2]*len(pred_s_list[b])
        idx = -1
        for pred_s,pred_v in zip(pred_s_list[b],pred_v_list[b]):
            idx+=1
            try:
                root_p = generate_ast(pred_s,pred_v)
                pred = data_tools.ast2command(root_p,loose_constraints=False) 
                preds.append(pred)
                pred_indices.append(idx)  
            except Exception as e:
                pred_score[idx] = -2
                continue

        max_score = -1
        inner_scores = []
        min_above_zero = False
        for idx,p in zip(pred_indices,preds):
            score = compute_metric(p,1,truth,{'u1':1.0,'u2':1.0})
            pred_score[idx] = score
            inner_scores.append(score)
            if score>0:
                min_above_zero = True
            max_score = max(score,max_score)
        if min_above_zero:        
            scores.append(max_score)
        else:
            if len(inner_scores)!=0:
                av = sum(inner_scores)/len(inner_scores)
                scores.append(av)
            else:
                scores.append(-1)

        all_scores.append(pred_score)
        if len(preds)==0:
            c+=1
            
    scores = np.array(scores)
    all_scores = np.array(all_scores)
    return (scores,all_scores), c