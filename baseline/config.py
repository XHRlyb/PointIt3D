import os

class Config:
    dataset_path = '..\ScanNet_with_eric'
    labels_dir = os.path.join(dataset_path, 'scannetv2-labels.combined.tsv')
    seg_suffix = '_vh_clean_2.0.010000.segs.json'
    agg_suffix = '.aggregation.json'
    ply_suffix = '_with_eric.ply'
    ans_name = 'answer.txt'
    assert os.path.exists(dataset_path), dataset_path + 'not found'

    seed = 42
    ang = 5
    batch_size = 16
    max_epoch = 20
    save_epoch = 1
    test_epoch = 1
    display_steps = 10
    
    optimizer = 'adam' # or sgd
    lr = 1e-4
    dropout = 0.5
    pool_dropout = 0.5
    weight_decay = 5e-5

    exp_root = os.path.join(os.getcwd(), './exps/')
    prefix = 'demo'
    exp_name = 'demo'
    exp_path = os.path.join(exp_root, prefix)
    while os.path.exists(exp_path):
        idx = os.path.basename(exp_path).split(prefix)[-1]
        try:
            idx = int(idx) + 1
        except:
            idx = 1
        exp_name = prefix + str(idx)
        exp_path = os.path.join(exp_root, exp_name)
    print('Exps name {}'.format(os.path.basename(exp_path)))
    chkpt_dir = os.path.join(exp_path, 'chkpts')
    log_dir = os.path.join(exp_path, 'logs')

    def create_path(self):
        print('Creating exp dir: ', self.exp_path)
        os.makedirs(self.exp_path)
        os.makedirs(self.chkpt_dir)
        os.makedirs(self.log_dir)

    


    