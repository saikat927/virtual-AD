import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from image_processing import process_image, show_image_pair, convert_tensor, warp_image, visualize_images, viz, process_edge_image
from networks import OffsetNetV2
import torch.optim as optim
from losses import iou, hausdorff_distance, weighted_iou

w, h = 256, 256

# read frame
frame = cv2.imread('../0.jpg',0)
frame_bin, frame_dist = process_image(frame, w, h)
frame_bin_tensor = convert_tensor(frame_bin)
frame_dist_tensor = convert_tensor(frame_dist)
#show_image_pair(frame_bin, frame_dist)

# read field
field = cv2.imread('../field1.png',0)
field_bin, field_dist = process_image(field, w, h)
field_bin_tensor = convert_tensor(field_bin)
field_dist_tensor = convert_tensor(field_dist)
# #show_image_pair(field_bin, field_dist)

initial_homography = torch.tensor([[ 2.57041223e+00 ,-9.57286500e-01,  8.82476624e+01],
 [ 4.00910643e-01 , 6.22470662e-01, -3.64845922e+01],
 [-5.33360436e-04, -1.02989403e-04,  1.00000000e+00]])

# Flatten the initial homography to a 1D tensor with 8 values
initial_homography_flat = torch.cat((initial_homography[:2].view(-1), initial_homography[2, :2]), dim=0)


initial_homography.requires_grad = True
old_homography = initial_homography.clone().detach()

# Define an optimizer (e.g., Adam)
optimizer = optim.Adam([initial_homography], lr=1e-4)

# Initialize the offset network
offset_net = OffsetNetV2()
mse = torch.nn.MSELoss()

# Gradient descent loop
for epoch in range(100):
    optimizer.zero_grad()

     # Predict the offsets using the offset network
    offsets = offset_net(initial_homography_flat.unsqueeze(0))
    
    # Add the predicted offsets to the initial homography matrix
    predicted_homography = initial_homography + offsets.squeeze(0)
    
    # Apply homography transformation (pseudo-code, replace with actual implementation)
    field_bin_warped = warp_image(field_bin_tensor.float(), predicted_homography.unsqueeze(0))
    field_dist_warped = warp_image(field_dist_tensor.float(), predicted_homography.unsqueeze(0))
    
    mse_loss = mse(frame_bin_tensor.float()*field_dist_warped, field_bin_tensor.float()*field_dist_warped)
    norm_loss = torch.norm((frame_dist_tensor-field_dist_warped)*frame_dist_tensor)/(torch.norm(frame_dist_tensor)+.00000001)
    iou_loss = iou(frame_bin_tensor, field_bin_warped)
    #hausdorff_loss = hausdorff_distance(frame_bin_tensor, field_bin_warp)
    # Total loss
    total_loss = .4*norm_loss + .6*iou_loss #+ hausdorff_loss 

    
    # Backpropagation
    total_loss.backward()
    
    # Update the homography matrix
    optimizer.step()
    
    if epoch % 1 == 0:
        print(f'Epoch {epoch}, Mse Loss: {mse_loss}, Norm Loss: {norm_loss}, iou Loss: {iou_loss}, total: {total_loss}')
        viz(frame_bin_tensor, field_bin_warped, iteration = epoch)
        

    # Check for convergence
    if total_loss.item() < .005:
        print('Reached threshold.. breaking')
        break

print('the final homography matrix is', initial_homography+offsets.squeeze(0))
viz(frame_bin_tensor, field_bin_warped,0)