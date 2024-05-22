import sys
sys.path.append('core')
import argparse
import ast
import numpy as np
import cv2
import open3d as o3d
import re
import os
import glob
import torch
from tqdm import tqdm
from pathlib import Path
from igev_stereo import IGEVStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import rbd
from lightglue.utils import numpy_image_to_torch

DEVICE = 'cuda'

class Intrinsics:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
def parse_coordinates(coord):
    try:
        # Use ast.literal_eval to safely evaluate the string as a list
        return ast.literal_eval(coord)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Coordinates must be a comma-separated list of numbers inside brackets.")

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def match_points(image0,image1) -> np.array:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
    
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
    matcher = LightGlue(features="superpoint").eval().to(device)
    image0 = numpy_image_to_torch(image0)
    image1 = numpy_image_to_torch(image1)

    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    
    m_kpts0_numpy = m_kpts0.cpu().detach().numpy()
    m_kpts1_numpy = m_kpts1.cpu().detach().numpy()
    
    return m_kpts0_numpy,m_kpts1_numpy

        
def read_camera_intrinsics(args) -> list:
    with open(args.f) as file:
        content = file.read()
    fx_match = re.search(r"fx = ([0-9.]+)", content)
    fy_match = re.search(r"fy = ([0-9.]+)", content)
    cx_match = re.search(r"cx = ([0-9.]+)", content)
    cy_match = re.search(r"cy = ([0-9.]+)", content)
    baseline_match = re.search(r"baseline = ([0-9.]+)", content)

    fx_value = float(fx_match.group(1))
    fy_value = float(fy_match.group(1))
    cx_value = float(cx_match.group(1))
    cy_value = float(cy_match.group(1))
    #baseline in millimeters
    baseline = 1000 * float(baseline_match.group(1))
    
    intrinsics_array = list([Intrinsics(fx_value,fy_value,cx_value,cy_value),
                             Intrinsics(fx_value,fy_value,cx_value,cy_value),baseline])
    return(intrinsics_array)
      
def demo(args):
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)
    
    #define camera intrinsics
    intrinsics_list = read_camera_intrinsics(args)
    camera_intrinsics_l = intrinsics_list[0] 
    camera_intrinsics_r = intrinsics_list[1]  
    baseline = intrinsics_list[2]   

    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))
    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(str(args.output_directory)+"/pointCloud")
    output_directory.mkdir(exist_ok=True)
    
    base_image_nparray = np.array
    base_number_points = int
    base_point_cloud_points = np.array
    average_distance = []
    matched_percentage = []

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for i, (imfile1, imfile2) in enumerate(tqdm(list(zip(left_images, right_images)))):            
            image1_filename = os.path.basename(imfile1)
            image1_nparray = np.array(Image.open(imfile1)).astype(np.uint8)
            
            height, width = image1_nparray.shape[:2]

            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            disparity = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disparity = padder.unpad(disparity).squeeze()
            disparity = disparity.cpu().numpy().squeeze()

            file_stem = imfile1.split('/')[-2]
            if args.save_numpy:
                np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
            #plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap='jet')

            #min_disparity = np.min(disparity)

            #Generate depth image and display it as grey scale
            np_depth_image = np.array(camera_intrinsics_l.fx*baseline/np.absolute(disparity+(camera_intrinsics_r.cx-camera_intrinsics_l.cx)))   
            
            #uint_img = np.array(disparity*255/min_disparity).astype('uint8')
            #grey_image = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)

            #visualizing grey scale depth image
            #cv2.imwrite(str(output_directory / f"{file_stem}_greyscale.png"), grey_image)
            #cv2.imshow("disparity", grey_image)
            #cv2.waitKey(0)
                       
            #o3d_depth_image = o3d.geometry.Image(np_depth_image.astype(np.float32))
            o3d_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height,
                                                                    camera_intrinsics_l.fx, camera_intrinsics_l.fy,
                                                                    camera_intrinsics_l.cx, camera_intrinsics_l.cy)
            coordinates_0 = args.first_point
            coordinates_1 = args.second_point
            
            image1_nparray[coordinates_0[1],coordinates_0[0]] = [255,0,0]
            image1_nparray[coordinates_1[1],coordinates_1[0]] = [255,0,0]
            
            color_image = o3d.geometry.Image(image1_nparray)
            depth_image = o3d.geometry.Image(np_depth_image)
            
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image, depth_trunc=float('inf'),convert_rgb_to_intensity=False)

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_camera_intrinsic)
            #pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth_image, o3d_camera_intrinsic)
            
            pcd_points = np.asarray(pcd.points)
            o3d.io.write_point_cloud(str(output_directory / f"{image1_filename}_pointcloud.pcd"), pcd, write_ascii=False, compressed=False, print_progress=False)
                         
            tip_coordinates = pcd_points[coordinates_0[0] + coordinates_0[1]*width]
            end_coordinates = pcd_points[coordinates_1[0] + coordinates_1[1]*width]
            distance = euclidean(tip_coordinates, end_coordinates)
            print(distance)
            
            with open("circuit_length.txt", "a") as file:
                file.write(str(image1_filename) + ": " + str(distance) + "\n")
            with open("coordinates.txt", "a") as file:
                file.write(str(image1_filename) + ": " + str(coordinates_0) + "; " + str(coordinates_1) + "\n")
            #visualization
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name='pointcloud', width=1000, height=1000)
            vis.add_geometry(pcd)
            vis.run()
            vis.destroy_window()
            
    # save the data for each video
    
    
    #plt.figure()
    #plt.plot(indices, average_distance)
    #plt.xlabel('Time Frame')
    #plt.ylabel('Average Distance')
    #plt.title('Plot 1')

    #plt.figure()  # Create another new figure
    #plt.scatter(average_distance, matched_percentage)
    #plt.xlabel('Average Distance')
    #plt.ylabel('Matched Percentage')
    #plt.title('Plot 2')


    #plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-f', help="file path of camera intrinsics", required=True)
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", required=True)
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", required=True)
    parser.add_argument('--first_point', type=parse_coordinates, help="coordinate 1 (format: [x, y])", required=True)
    parser.add_argument('--second_point', type=parse_coordinates, help="coordinate 2 (format: [x, y])", required=True)
    parser.add_argument('--output_directory', help="directory to save output", default="pointcloud_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    
    args = parser.parse_args()
    demo(args)
