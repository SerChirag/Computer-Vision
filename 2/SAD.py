import numpy as np
from PIL import Image
import cv2 as cv2

def stereo_match(left_img, right_img, kernel, max_offset):
    # Load in both images, assumed to be RGBA 8bit per channel images
    left_img = Image.open(left_img).convert('L')
    left = np.asarray(left_img)
    print(left.shape)
    right_img = Image.open(right_img).convert('L')
    right = np.asarray(right_img)    
    w, h = left_img.size  # assume that both images are same size   
    
    # Depth (or disparity) map
    depth = np.zeros((w, h), np.uint8)
    depth.shape = h, w
       
    kernel_half = int(kernel / 2)    
    offset_adjust = 255 / max_offset  # this is used to map depth map output to 0-255 range
      
    for y in range(kernel_half, h - kernel_half):      
        # print(".", end="", flush=True)  # let the user know that something is happening (slowly!)
        
        for x in range(kernel_half, w - kernel_half):
            best_offset = 0
            prev_sad = 65534
            
            for offset in range(max_offset):               
                sad = 0
                sad_temp = 0                            
                
                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):
                        sad_temp = abs(int(left[y+v, x+u]) - int(right[y+v, (x+u) - offset]))  
                        sad += sad_temp          

                # if this value is smaller than the previous ssd at this block
                # then it's theoretically a closer match. Store this value against
                # this block..
                if sad < prev_sad:
                    prev_sad = sad
                    best_offset = offset
                            
            # set depth output for this x,y location to the best match
            depth[y, x] = best_offset 
                                
    # Convert to PIL and save it
    norm_image = cv2.normalize(depth, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255
    cv2.imwrite('depth_SAD_' + str(kernel) + "_" + str(max_offset) + '.png',norm_image)

if __name__ == '__main__':
    stereo_match("images/1.ppm", "images/2.ppm", 6, 30)  # 6x6 local search kernel, 30 pixel search range


