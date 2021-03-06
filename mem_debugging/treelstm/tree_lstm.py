"""
PyTorch Child-Sum Tree LSTM model

See Tai et al. 2015 https://arxiv.org/abs/1503.00075 for model description.

[IMPLEMENTATION ADAPTED FROM https://github.com/unbounce/pytorch-tree-lstm]
"""

import torch


### MEMORY DEBUGGING
import os
import psutil

def mem_diff(prev_mem, legend=0, print_mem=False):
    current_mem = get_current_mem()
    mem_change = current_mem - prev_mem
    if mem_change and print_mem:
        print(f'\n\n CPU memory difference {legend}: \t\t {mem_change}MB \n')
    return current_mem, mem_change

def get_current_mem():
    """
    Get memory usage in MB for current process

    Returns
    -------
    float
        MBs of memory used by current process
    """
    conversion_rate = 2**20 # CONVERT TO MB
    pid = os.getpid()
    proc = psutil.Process(pid)
    # mem_gb = "{:.2f}".format(proc.memory_info()[0] / conversion_rate)
    return proc.memory_info()[0] / conversion_rate
### / MEMORY DEBUGGING


class TreeLSTM(torch.nn.Module):
    '''PyTorch TreeLSTM model that implements efficient batching.
    '''
    def __init__(self, in_features, out_features, word_emb_dim):
        '''TreeLSTM class initializer

        Takes in int sizes of in_features and out_features and sets up model Linear network layers.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.word_emb_dim = word_emb_dim

        # @DR
        self.word_embedding = torch.nn.Embedding(in_features, word_emb_dim)

        # bias terms are only on the W layers for efficiency
        ## UNCOMMENT FOR ORIGINAL IMPLEMENTATION
        # self.W_iou = torch.nn.Linear(self.in_features, 3 * self.out_features)
        # @DR
        self.W_iou = torch.nn.Linear(self.word_emb_dim, 3 * self.out_features)
        self.U_iou = torch.nn.Linear(self.out_features, 3 * self.out_features, bias=False)

        # f terms are maintained seperate from the iou terms because they involve sums over child nodes
        # while the iou terms do not
        ## UNCOMMENT FOR ORIGINAL IMPLEMENTATION
        # self.W_f = torch.nn.Linear(self.in_features, self.out_features)
        # @DR
        self.W_f = torch.nn.Linear(self.word_emb_dim, self.out_features)
        self.U_f = torch.nn.Linear(self.out_features, self.out_features, bias=False)
        # print(f'W_f size {self.word_emb_dim}x{self.out_features}')
        # print(f'U_f size {self.out_features}x{self.out_features}')

    def forward(self, mem_changes, features, node_order, adjacency_list, edge_order):
        '''Run TreeLSTM model on a tree data structure with node features

        Takes Tensors encoding node features, a tree node adjacency_list, and the order in which 
        the tree processing should proceed in node_order and edge_order.
        '''

        # Total number of nodes in every tree in the batch
        batch_size = node_order.shape[0]

        # Retrive device the model is currently loaded on to generate h, c, and h_sum result buffers
        device = next(self.parameters()).device
        
        last_mem, mem_change  = mem_diff(get_current_mem(), legend="**IN ENCODER** before h and c (#4.3.1)", print_mem=False) # MEM DEBUGGING!!!
        if mem_change: mem_changes['bf_hc_431'].append(mem_change)

        # h and c states for every node in the batch
        h = torch.zeros(batch_size, self.out_features, device=device)
        c = torch.zeros(batch_size, self.out_features, device=device)

        last_mem, mem_change  = mem_diff(last_mem, legend="**IN ENCODER** after h and c (#4.3.2)", print_mem=False) # MEM DEBUGGING!!!
        if mem_change: mem_changes['af_hc_432'].append(mem_change)

        # populate the h and c states respecting computation order
        for n in range(node_order.max() + 1):
            self._run_lstm(mem_changes, n, h, c, features, node_order, adjacency_list, edge_order)
        
        # print('TreeLSTM batch size', batch_size)
        # print('TreeLSTM h size', h.shape)
        # print('TreeLSTM c size', c.shape)

        return h, c

    def _run_lstm(self, mem_changes, iteration, h, c, features, node_order, adjacency_list, edge_order):
        '''Helper function to evaluate all tree nodes currently able to be evaluated.
        '''
        # N is the number of nodes in the tree
        # n is the number of nodes to be evaluated on in the current iteration
        # E is the number of edges in the tree
        # e is the number of edges to be evaluated on in the current iteration
        # F is the number of features in each node
        # M is the number of hidden neurons in the network

        # node_order is a tensor of size N x 1
        # edge_order is a tensor of size E x 1
        # features is a tensor of size N x F
        # adjacency_list is a tensor of size E x 2

        # node_mask is a tensor of size N x 1
        node_mask = node_order == iteration
        # edge_mask is a tensor of size E x 1
        edge_mask = edge_order == iteration

        # x is a tensor of size n x F
        # print('+++++++++++++++ Features argmax ', torch.argmax(features[node_mask, :], dim=1))
        ## UNCOMMENT FOR ORIGINAL IMPLEMENTATION
        # x = features[node_mask, :]
        
        # @DR
        # DEBUGGING: DIMENSION CHECK
        # print(f'features \t {features} \n')#.size()[0]
        # print(f'node_mask \t {node_mask} \n')#.size()[0]
        # print(f'features[node_mask, :] \t {features[node_mask]}')#.size()[0]

        # @DR: OLD-REMOVE
        # DIMENSIONALITY FIX, IF features[parent_indexes, :] HAS
        # A SINGLE EXAMPLE TAKE THE GLOBAL ARGMAX AND UNSQUEEZE,
        # ELSE GET THE ARGMAX FROM EACH EXAMPLE
        # if features[node_mask, :].size()[0] == 1:
        #     x = self.word_embedding(torch.argmax(features[node_mask, :])).unsqueeze(0)
        # else:
        #     # print('@@@ features[node_mask, :].size()[0] larger than 1')
        #     # print(f'torch.argmax(features[node_mask, :]) {torch.argmax(features[node_mask, :], dim=1)}')
        #     x = self.word_embedding(torch.argmax(features[node_mask, :], dim=1))

        x = self.word_embedding(features[node_mask])
        # print(f'######## x: \t {x}')

        last_mem, mem_change  = mem_diff(get_current_mem(), legend="**IN TREELSTM** after word embeddings (#4.3.2.1)", print_mem=False) # MEM DEBUGGING!!!
        if mem_change: mem_changes['w_embs_4321'].append(mem_change)

        # print(f'iteration: {iteration} \n \t node mask: {node_mask} \t x: {x}')
        # print(f'x length {x.size()}')

        # At iteration 0 none of the nodes should have children
        # Otherwise, select the child nodes needed for current iteration
        # and sum over their hidden states
        if iteration == 0:
            iou = self.W_iou(x)
            last_mem, mem_change  = mem_diff(get_current_mem(), legend="**IN TREELSTM** it-0 (#4.3.2.1.1)", print_mem=False) # MEM DEBUGGING!!!
            if mem_change: mem_changes['it_0_43211'].append(mem_change)
        else:
            # adjacency_list is a tensor of size e x 2
            adjacency_list = adjacency_list[edge_mask, :]

            last_mem, mem_change  = mem_diff(get_current_mem(), legend="**IN TREELSTM** after word embeddings (#4.3.2.1.2)", print_mem=False) # MEM DEBUGGING!!!
            if mem_change: mem_changes['adj_msk_43212'].append(mem_change)
    
            # parent_indexes and child_indexes are tensors of size e x 1
            # parent_indexes and child_indexes contain the integer indexes needed to index into
            # the feature and hidden state arrays to retrieve the data for those parent/child nodes.
            parent_indexes = adjacency_list[:, 0]
            child_indexes = adjacency_list[:, 1]

            last_mem, mem_change  = mem_diff(get_current_mem(), legend="**IN TREELSTM** parent ixs (#4.3.2.1.3)", print_mem=False) # MEM DEBUGGING!!!
            if mem_change: mem_changes['pnt_ixs_43213'].append(mem_change)

            # child_h and child_c are tensors of size e x 1
            # CORRECTION: child_h and child_c are tensors of size e x hid_dim
            child_h = h[child_indexes, :]
            child_c = c[child_indexes, :]

            last_mem, mem_change  = mem_diff(get_current_mem(), legend="**IN TREELSTM** child h c (#4.3.2.1.4)", print_mem=False) # MEM DEBUGGING!!!
            if mem_change: mem_changes['chld_hc_43214'].append(mem_change)

            # print(f'adjacency_list: {adjacency_list}')

            # Add child hidden states to parent offset locations
            ### UNCOMMENT AND CHANGE BACK TO torch.unique_consecutive (ORIGINAL IMPLEMENTATION)
            _, child_counts = torch.unique_consecutive(parent_indexes, return_counts=True)
            child_counts = tuple(child_counts)

            last_mem, mem_change  = mem_diff(get_current_mem(), legend="**IN TREELSTM** child counts (#4.3.2.1.5)", print_mem=False) # MEM DEBUGGING!!!
            if mem_change: mem_changes['chld_counts_43215'].append(mem_change)

            # print(f'child_counts {child_counts}')

            parent_children = torch.split(child_h, child_counts)
            parent_list = [item.sum(0) for item in parent_children]

            last_mem, mem_change  = mem_diff(get_current_mem(), legend="**IN TREELSTM** parent child (#4.3.2.1.6)", print_mem=False) # MEM DEBUGGING!!!
            if mem_change: mem_changes['prnt_chld_43216'].append(mem_change)

            # print(f'parent_children {parent_children}')
            # print(f'parent_list {parent_list}')

            h_sum = torch.stack(parent_list)
            iou = self.W_iou(x) + self.U_iou(h_sum)
            
            last_mem, mem_change  = mem_diff(get_current_mem(), legend="**IN TREELSTM** h stack (#4.3.2.1.7)", print_mem=False) # MEM DEBUGGING!!!
            if mem_change: mem_changes['h_stack_43217'].append(mem_change)

        last_mem, mem_change  = mem_diff(last_mem, legend="**IN TREELSTM** adj lists (#4.3.2.2)", print_mem=False) # MEM DEBUGGING!!!
        if mem_change: mem_changes['adj_lists_4322'].append(mem_change)

        # i, o and u are tensors of size n x M
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)

        last_mem, mem_change  = mem_diff(last_mem, legend="**IN TREELSTM** IOU (#4.3.2.3)", print_mem=False) # MEM DEBUGGING!!!
        if mem_change: mem_changes['iou_4323'].append(mem_change)

        # At iteration 0 none of the nodes should have children
        # Otherwise, calculate the forget states for each parent node and child node
        # and sum over the child memory cell states
        if iteration == 0:
            c[node_mask, :] = i * u
        else:
            # f is a tensor of size e x M
            # print('features[parent_indexes, :].size()', features[parent_indexes, :].size())
            # print('child_h.size()', child_h.size())
            # f = self.W_f(features[parent_indexes, :]) + self.U_f(child_h)
            # print(f'features[parent_indexes, :] {features[parent_indexes, :]}')
            # print(f'torch.argmax(features[parent_indexes, :]) {torch.argmax(features[parent_indexes, :], dim=1)}')

            # @DR: OLD-REMOVE
            # DIMENSIONALITY FIX, IF features[parent_indexes, :] HAS
            # A SINGLE EXAMPLE TAKE THE GLOBAL ARGMAX AND UNSQUEEZE,
            # ELSE GET THE ARGMAX FROM EACH EXAMPLE
            # if features[parent_indexes, :].size()[0] == 1:
            #     x_parents = self.word_embedding(torch.argmax(features[parent_indexes, :])).unsqueeze(0)
            # else:
            #     x_parents = self.word_embedding(torch.argmax(features[parent_indexes, :], dim=1))

            x_parents = self.word_embedding(features[parent_indexes])

            # print('x_parents type', x_parents.type())
            # print('x_parents', x_parents)
            f = self.W_f(x_parents) + self.U_f(child_h)
            f = torch.sigmoid(f)

            last_mem, mem_change  = mem_diff(last_mem, legend="**IN TREELSTM** after F (#4.3.2.4)", print_mem=False) # MEM DEBUGGING!!!
            if mem_change: mem_changes['after_f_4324'].append(mem_change)

            # fc is a tensor of size e x M
            fc = f * child_c

            # Add the calculated f values to the parent's memory cell state
            parent_children = torch.split(fc, child_counts)
            parent_list = [item.sum(0) for item in parent_children]

            last_mem, mem_change  = mem_diff(last_mem, legend="**IN TREELSTM** after parents (#4.3.2.5)", print_mem=False) # MEM DEBUGGING!!!
            if mem_change: mem_changes['after_parents_4325'].append(mem_change)

            c_sum = torch.stack(parent_list)
            c[node_mask, :] = i * u + c_sum
        
        last_mem, mem_change  = mem_diff(last_mem, legend="**IN TREELSTM** after C (#4.3.2.6)", print_mem=False) # MEM DEBUGGING!!!
        if mem_change: mem_changes['after_c_4326'].append(mem_change)

        h[node_mask, :] = o * torch.tanh(c[node_mask])

        # print(f'h: {h}')
