import os
import json
import requests
import argparse
import tqdm
import time
import torchvision

def fetch_and_dump(image_info, path_name='raw/train'):
    url = image_info['url']
    image_id = str(image_info['imageId'])

    os.makedirs(path_name, exist_ok=True)
    image_path = os.path.join(path_name, image_id + '.png')

    if os.path.exists(image_path):
        try:
            torchvision.io.read_image(image_path)
            return
        except:
            print("corrupted image, redownloading")
            pass
    
    while True:
        try:
            img_data = requests.get(url).content
            break
        except:
            print(f"Error fetching {url}, sleepting for 1 seconds")
            time.sleep(1)
            
        
    with open(image_path, 'wb') as handler:
        handler.write(img_data)

def read_json(path, jobid):
    f = open(path, 'r')
    data = json.load(f)
    f.close()

    images = data['images']

    # 16 jobs will run in parallel
    # each job will fetch 1/16 of the data
    print(f"*********** path: {path} ***********")
    print(len(images))
    low = int(len(images) / 16 * jobid)
    high = min(len(images), int(len(images) / 16 * (jobid + 1)))
    
    print(low, high)

    for image_info in tqdm.tqdm(images[low:high]):
        fetch_and_dump(image_info, path_name=path.replace('.json', ''))

def main(args):
    path = args.path
    jobid = args.jobid


    read_json(path=path, jobid=jobid)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--path', type=str, default='raw/train.json')
    args.add_argument('--jobid', type=int, choices=range(16))

    args = args.parse_args()

    main(args)
