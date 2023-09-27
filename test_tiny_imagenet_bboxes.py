import glob
import os
import skimage.io
def get_bbox_for_impath(impath):
    dataset_and_split = os.path.dirname(os.path.dirname(os.path.dirname(impath)))
    imname = os.path.basename(impath)
    synset = imname.split('_')[0]
    synset_path = os.path.join(dataset_and_split,synset)
    bboxes_path = os.path.join(synset_path,f'{synset}_boxes.txt')


    bboxes = parse_bboxes(bboxes_path)
    return bboxes
def parse_bboxes(file_path):
    bboxes = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into its components
            parts = line.strip().split()
            
            # Extract filename and bounding box coordinates
            image_filename = parts[0]
            bbox = {
                'x1': int(parts[1]),
                'y1': int(parts[2]),
                'x2': int(parts[3]),
                'y2': int(parts[4])
            }
            
            # Store the bounding box in the dictionary
            bboxes[image_filename] = bbox
    
    return bboxes


def test():
    dataset_folder = '/root/evaluate-saliency-4/generative-attribution-methods/large_faiss/tiny-imagenet-200'
    split = 'train'

    if split == 'train':
        
        filelist = glob.glob(os.path.join(dataset_folder,split,'*','images','*.JPEG'))
    elif split == 'test':

        
        filelist = glob.glob(os.path.join(dataset_folder,split,'images','*.JPEG'))
    filelist = list(filelist)
    i = 0
    impath = filelist[i]
    
    im = skimage.io.imread(impath)
    imname = os.path.basename(impath)
    bboxes = get_bbox_for_impath(impath)
    assert imname in bboxes
    # assert False
    '''
    import cam_benchmark.ground_truth_handler as ground_truth_handler
    bbox_info,target_ids,classnames = ground_truth_handler.get_gt(imroot,'imagenet')
    # bbox = cam_benchmark.imagenet_localization_parser()
    '''
test()