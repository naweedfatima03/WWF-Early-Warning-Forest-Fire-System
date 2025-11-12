import os
import sys
import getopt
import json

yolo_args = {
    'dest': None, # where the results will be stored 
    'weights': None, # the directory for the weights file
    'conf':None, # the confidence interval for detections
    'img_size':None, # the images will be resized to this shape
    'source':None # where the images will be read from
}

ptz_args = {
    'dest':None, # where the ptz camera will store images
    'num_imgs':None # the number of images that will be read in a rotation
}

def sort_args(opts,args):
    for opt, arg in opts:
        if opt == '-h':
            print ('main.py -w <weightsfile> -c <confidenceinterval> -i <imagesize> -s <imagesdirectory> -n <num-imgs> -d <resultsdirectory>')
            print ('main.py --weights <weightsfile> --conf <confidenceinterval> --img-size <imagesize> --source <imagesdirectory> --num-imgs <num-imgs> --dest <resultsdirectory>')
            sys.exit()
        elif opt =="--weights" or opt =="-w":
            yolo_args['weights']  = arg
        elif opt =="--dest" or opt =="-d":
            yolo_args['dest'] = arg
        elif opt =="--conf" or opt =="-c":
            yolo_args['conf'] = arg
        elif opt =="--source" or opt =="-s":
            yolo_args['source'] = arg
            ptz_args['dest'] = arg
        elif opt =="--img-size" or opt =="-i":
            yolo_args['img_size'] = arg
        elif opt =="--num-imgs" or opt =="-n":
            ptz_args['num_imgs'] = arg
        else:
            print(opt,'is not a valid argument')
            sys.exit()
            
    # Specify default arguments here
    if yolo_args['conf'] == None:
        yolo_args['conf'] = 0.5
        # print('--conf interval not given')
        # sys.exit()
    if yolo_args['img_size'] == None:
        yolo_args['img_size'] = 640
        # print('--img-size not specified')
        # sys.exit()
    if yolo_args['source'] == None:
        yolo_args['source'] = 'yolo/Dataset/test/images/'
        ptz_args['dest'] = 'yolo/Dataset/test/images/'
        # print('--source directory not given')
        # sys.exit()
    if yolo_args['dest'] == None:
        yolo_args['dest'] = 'exp/'
        # print('--dest directory not given')
        # sys.exit()
    if yolo_args['weights'] == None:
        yolo_args['weights'] = 'yolo/tiny_yolov7.pt'
        # print('--weights directory not given')
        # sys.exit()
    if ptz_args['num_imgs'] == None:
        ptz_args['num_imgs'] = 10
        # print('--number of images to be collected by ptz')
        # sys.exit()
        
    return

def ptz_images():
    os.system(
        "python3 ptz_camera/my_capture.py --dest " + str(ptz_args['dest']) + 
        " --num-imgs " + str(ptz_args['num_imgs'])
    )
    return

def yolo_detect():
    os.system("echo Hello")
    print("python3 yolo/detect.py --weights " + str(yolo_args['weights'])+ 
        " --conf " + str(yolo_args['conf']) + 
        " --img-size " + str(yolo_args['img_size']) +
        " --source " + str(yolo_args['source']) +
        " --name " + str(yolo_args['dest']) +
        " --save-txt" )
    # os.system("cd yolo")
    # os.system("python3 detect.py")
    os.system(
        "python3 yolo/detect.py" 
        "--weights " 
        + str(yolo_args['weights']) +
        " --conf " + str(yolo_args['conf']) + 
        " --img-size " + str(yolo_args['img_size']) +
        " --source " + str(yolo_args['source']) +
        " --name " + str(yolo_args['dest']) +
        " --save-txt" 
    )
    return

def isFire(img, json_dir):
    file_dir = json_dir + img + '.json'
    with open(file_dir) as json_file:
        data = json.load(json_file)
    count = int(data['count'])
    if count > 0:
        for det in range(count):
            class_indx = int(data['detections'][str(det)]['class'])
            if class_indx == 1:
                return True
        return False
    else: return False
    
def isSmoke(img, json_dir):
    file_dir = json_dir + img + '.json'
    with open(file_dir) as json_file:
        data = json.load(json_file)
    count = int(data['count'])
    if count > 0:
        for det in range(count):
            class_indx = int(data['detections'][str(det)]['class'])
            if class_indx == 0:
                return True
        return False
    else: return False

def isDetect(img, json_dir):
    file_dir = json_dir + img + '.json'
    with open(file_dir) as json_file:
        data = json.load(json_file)
    if int(data['count']) > 0:
        return True
    else: return False

def process_results(image_dir):
    imgs = os.listdir(image_dir)
    json_dir = 'yolo/runs/detect/exp4/json/'
    save_imgs = [] # contains imgs with detections
    for img in imgs:
        detected = isDetect(img, json_dir)
        if detected: save_imgs.append(img)
    print('Total number of images found:', len(imgs))
    print('Number of images with detections:', len(save_imgs))
    return

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], "hw:c:i:s:n:d:", 
                               ["weights=","conf=","img-size=","source=","num-imgs=","dest="])
    sort_args(opts, args)
    
    
    # get images from ptz cameras
    # ptz_images()
    
    # run yolov7 on the images in source directory
    yolo_detect()
    
    # check if there were detections
    # process_results(yolo_args['source'])
    
    
    