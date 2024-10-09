
import math
import decoder
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import matplotlib.patches as patches
from skimage.metrics import structural_similarity as ssim
from matplotlib.colors import ListedColormap, BoundaryNorm
from skimage.metrics import peak_signal_noise_ratio as psnr

# Y Only Video Codec
class Y_Video_codec:

    def __init__(self, h_pixels, w_pixels, frames, block_size, search_range, Qp, intra_dur, intra_mode, lam=None, VBSEnable=False, nRefFrames=1, yuv_file=None,  y_only_frame_arr=None, fast_me=False, FMEEnable=False, targetBR=0, row_budget=0, RCflag=False,frame_budget=0):
        self.y_only_f_arr      = None
        self.y_only_f_arr_blks = None
        self.y_only_f_arr_stch = None
        self.h_pixels          = h_pixels
        self.w_pixels          = w_pixels
        self.frames            = frames
        self.num_blks_per_f    = None
        self.num_blk_r         = None
        self.num_blk_c         = None
        self.blockified_f      = False
        self.stiched_f         = False
        self.block_size        = block_size
        self.sub_block_size    = block_size//2
        self.search_range      = search_range
        self.Qp                = Qp
        self.const_qp          = Qp
        self.Qpm1              = None
        self.intra_dur         = intra_dur
        self.intra_mode        = intra_mode
        self.Q                 = self.generate_Q_matrix(block_size, Qp)
        self.Qm1               = None
        self.decoder           = decoder.decoder(intra_mode, intra_dur, block_size, frames, h_pixels, w_pixels, Qp, nRefFrames, FMEEnable, lam, VBSEnable, RCflag=RCflag)
        self.encoded_package   = None
        self.encoded_package_f = False
        self.nRefFrames        = nRefFrames
        self.fast_me           = fast_me
        self.FMEEnable         = FMEEnable
        self.VBSEnable         = VBSEnable
        self.lam               = lam
        self.targetBR          = targetBR
        self.row_budget        = row_budget
        self.frame_budget      = frame_budget
        self.RCflag            = RCflag
        self.cif_qp_i_table   = np.array([139718.0, 132241.0, 108276.0, 84894.0, 63763.0, 45728.0, 31296.0, 20040.0, 12374.0, 7906.0, 6161.0, 5626.0])
        self.cif_qp_i_table/=8
        self.cif_qp_p_table    =np.array([117082.0, 90450.0, 68915.0, 50002.0, 34444.0, 24746.0, 17354.0, 12297.0, 7514.0, 4836.0, 4167.0, 3990.0])
        self.cif_qp_p_table/=8
        self.qcif_qp_i_table   = np.array([69422.0, 58143.0, 47524.0, 38942.0, 30895.0, 22056.0, 13663.0, 7194.0, 3931.0, 2198.0, 1461.0, 1305.0])
        self.qcif_qp_i_table/=8
        self.qcif_qp_p_table    =np.array([58183.0, 44377.0, 35931.0, 28604.0, 22263.0, 17282.0, 12167.0, 8099.0, 4509.0, 2697.0, 2074.0, 1969.0])
        self.qcif_qp_p_table/=8
        
        self.qp_table          = np.zeros((frames,h_pixels//block_size))
        if Qp > 0:
            self.Qpm1 = self.Qp-1
            self.Qm1  = self.generate_Q_matrix(self.sub_block_size, self.Qpm1)
        else:
            self.Qpm1 = self.Qp
            self.Qm1  = self.generate_Q_matrix(self.sub_block_size, self.Qpm1)

        if yuv_file != None:
            self.y_only_f_arr = self.read_yuv(yuv_file, h_pixels, w_pixels, frames)
        else:
            self.y_only_f_arr = y_only_frame_arr # numpy arr with index (frame, px height, px width)

    # Read Y-component from a YUV 4:2:0 file
    @staticmethod
    def read_yuv(raw_yuv_420_f, height, width, frames):
        size_y  = width * height # Num pixel for Y component
        size_uv = int(size_y/4)

        Y_ONLY = None           # Y Only frame array
        
        with open(raw_yuv_420_f, 'rb') as file:
            for i in range(frames):
                Y = np.frombuffer(file.read(size_y), dtype=np.uint8).reshape(1, height, width) # Read Y frame from file

                if i == 0: Y_ONLY = Y
                else:      Y_ONLY = np.vstack((Y_ONLY, Y))

                file.read(size_uv * 2) # Skip U and V framr from file
        
        return Y_ONLY
    
    # Padding a numpy array. Reference: https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros
    @staticmethod
    def pad(array, reference_shape, pad_with=None):
        arr_r, arr_c = array.shape
        
        if pad_with == None: result = np.zeros(reference_shape)
        else: result = np.zeros(reference_shape) + pad_with

        result[0:arr_r,0:arr_c] = array
        
        return result

    def pad_hw(self, array, i, pad_with=None):
        
        r_req        = math.ceil(self.h_pixels/i)
        c_req        = math.ceil(self.w_pixels/i)
        h_pixels_req = r_req * i
        w_pixels_req = c_req * i
        reference_shape = (h_pixels_req, w_pixels_req)

        arr_r, arr_c = array.shape
        
        if pad_with == None: result = np.zeros(reference_shape)
        else: result = np.zeros(reference_shape) + pad_with

        result[0:arr_r,0:arr_c] = array
        
        return result

    # Creating blocks out of the frame. Reference: https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
    @staticmethod
    def blockshaped(arr, nrows, ncols):
        h, w = arr.shape
        assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
        assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
        return (arr.reshape(h//nrows, nrows, -1, ncols)
                .swapaxes(1,2)
                .reshape(-1, nrows, ncols))

    @staticmethod
    def unblockshaped(arr, h, w):
        n, nrows, ncols = arr.shape
        return (arr.reshape(h//nrows, -1, nrows, ncols)
                .swapaxes(1,2)
                .reshape(h, w))

    # Converts each frame into blocks of size ixi in raster order
    def blockify_y_only_frames(self, i=None, replace=True):
        
        if i == None: i = self.block_size

        Y_ONLY_BLOCKS = None
        r_req        = math.ceil(self.h_pixels/i)
        c_req        = math.ceil(self.w_pixels/i)
        h_pixels_req = r_req * i
        w_pixels_req = c_req * i

        num_blocks = int((h_pixels_req*w_pixels_req)/(i*i)) 

        for j in range(0, self.y_only_f_arr.shape[0]): # Loop through all frames. Frames are less, hence normal for-loop is fine
            frame = self.y_only_f_arr[j]
            
            if self.h_pixels % i != 0 or self.w_pixels % i != 0:
                frame = self.pad(frame, (h_pixels_req, w_pixels_req), 128)
            
            frame = self.blockshaped(frame, i, i).reshape(1, num_blocks, i, i)

            if j == 0:
                Y_ONLY_BLOCKS = frame
            else:
                Y_ONLY_BLOCKS = np.vstack((Y_ONLY_BLOCKS, frame))

        if replace:
            self.y_only_f_arr_blks = Y_ONLY_BLOCKS
            self.num_blk_r = r_req
            self.num_blk_c = c_req
            self.num_blks_per_f = num_blocks
            self.blockified_f = True
            self.block_size = i

        return Y_ONLY_BLOCKS
    
    # Stitch together the blks
    def stitch_blks_y_only_frames(self, replace=True):
        Y_ONLY_BLOCKS_AVG = None
        for j in range(0, self.frames):
            frame_blk = self.y_only_f_arr_blks[j, :, :, :]
            frame = self.unblockshaped(frame_blk, self.h_pixels, self.w_pixels).reshape(1, self.h_pixels, self.w_pixels)
            if j == 0: Y_ONLY_BLOCKS_AVG = frame
            else: Y_ONLY_BLOCKS_AVG = np.vstack((Y_ONLY_BLOCKS_AVG, frame))

        if replace:
            self.stiched_f         = True
            self.y_only_f_arr_stch = Y_ONLY_BLOCKS_AVG
    
    def get_qp_val(self,qp_table, row_budget):
        # print('row==========',row_budget)
        # print(row_budget)
        # print(qp_table)
        for i,val in enumerate(qp_table):
            # print(i,val)
            if val<=row_budget:
                # print(i,val,row_budget)
                # exit()
                return i
        return 11
            
    def changed_qp(self,qp):
        self.Qp= qp
        self.Q = self.generate_Q_matrix(self.block_size, qp)
        if self.Qp > 0:
            self.Qpm1 = self.Qp-1
            self.Qm1  = self.generate_Q_matrix(self.sub_block_size, self.Qpm1)
        else:
            self.Qpm1 = self.Qp
            self.Qm1  = self.generate_Q_matrix(self.sub_block_size, self.Qpm1)        
        return
    # View frame
    def view_frame(self, frame):
        plt.imshow(self.y_only_f_arr[frame], cmap="gray")
        plt.show()

    # View blockified frame
    def view_blockified_frame(self, frame):
        if self.blockified_f == False:
            print("[ERROR] No blockified frames available!")
            return
        else:
            fig = plt.figure(figsize=(10, 7))
            r = self.num_blk_r
            c = self.num_blk_c

            for i in range(0, self.num_blks_per_f):
                fig.add_subplot(r, c, i+1)
                plt.imshow(self.y_only_f_arr_blks[frame][i], cmap="gray", vmin=0, vmax=255)

            plt.show()

    # View stitched frame
    def view_stitched_frame(self, frame):
        if self.stiched_f == False:
            print("[ERROR] No stitched frames available!")
            return
        else:
            plt.imshow(self.y_only_f_arr_stch[frame], cmap="gray", vmin=0, vmax=255)
            plt.show()
    
    # SSIM between stitched and original video
    def calc_ssim_stch_ori(self):
        frames = []
        ssim_scores = []

        for i in range(0, self.frames):
            frames.append(i)
            ssim_scores.append(ssim(self.y_only_f_arr[i], self.y_only_f_arr_stch[i]))
        
        plt.plot(frames, ssim_scores)
        plt.xlabel("Frames")
        plt.ylabel("SSIM")
        plt.title("SSIM for all frames")
        plt.show()

    # PSNR between stitched and original video
    def calc_psnr_stch_ori(self):
        frames = []
        psnr_scores = []

        for i in range(0, self.frames):
            frames.append(i)
            psnr_scores.append(psnr(self.y_only_f_arr[i], self.y_only_f_arr_stch[i]))

        plt.plot(frames, psnr_scores)
        plt.xlabel("Frames")
        plt.ylabel("PSNR")
        plt.title("PSNR for all frames")
        plt.show()
        
    # Average out all blocks
    def average_blocks(self, replace=True):
        if self.blockified_f == False:
            print("[ERROR] No blockified frames available!")
            return
        else:
            FRAMES = None
            for i in range(0, self.frames):
                BLOCKS = None
                for j in range(0, self.num_blks_per_f):
                    block = self.y_only_f_arr_blks[i,j,:,:]
                    block_avg = int(np.average(block))
                    block[block>-1] = block_avg
                    block = block.reshape(1, block.shape[0], block.shape[1])
                    
                    if j == 0: BLOCKS = block
                    else: BLOCKS = np.vstack((BLOCKS, block))

                BLOCKS = BLOCKS.reshape(1, BLOCKS.shape[0], BLOCKS.shape[1], BLOCKS.shape[2])

                if i == 0: FRAMES = BLOCKS
                else: FRAMES = np.vstack((FRAMES, BLOCKS))

        if replace:
            self.y_only_f_arr_blks = FRAMES
        
        return FRAMES

    #############################    
    # Motion Estimation functions
    #############################

    #Compute Mean Absolute Error between two blocks.
    def compute_mae(self, block1, block2):
        return np.mean(np.abs(block1 - block2))

    # # Find the best match for a block in a reference frame within a defined search range.
    # def find_best_match(self, current_block, ref_frame, x, y, block_size=None, search_range=None):

    #     if block_size == None: block_size = self.block_size
    #     if search_range == None: search_range = self.search_range
        
    #     best_mae = float('inf')
    #     best_mv = (0, 0)

    #     for dx in range(-search_range, search_range + 1):
    #         for dy in range(-search_range, search_range + 1):
    #             # Ensure the reference block is entirely within the frame boundaries
    #             if (x+dx >= 0 and x+dx+block_size <= ref_frame.shape[1] and
    #                 y+dy >= 0 and y+dy+block_size <= ref_frame.shape[0]):

    #                 ref_block = ref_frame[y+dy:y+dy+block_size, x+dx:x+dx+block_size]
    #                 mae = self.compute_mae(current_block, ref_block)

    #                 # Check if this block is a better match than the previous ones
    #                 if mae < best_mae:
    #                     # print(best_mae)      
    #                     best_mae = mae
    #                     best_mv = (dx, dy)

    #                 # If MAE is the same, check motion vector magnitude
    #                 elif mae == best_mae: 
    #                     if (abs(dx) + abs(dy)) < (abs(best_mv[0]) + abs(best_mv[1])) or (abs(dx) + abs(dy)) == (abs(best_mv[0]) + abs(best_mv[1])) and (dy < best_mv[1] or (dy == best_mv[1] and dx < best_mv[0])):
    #                         best_mv = (dx, dy)      
    #     return best_mv, best_mae

    # Perform block-based motion estimation for the current frame against a reference frame."""
    # def inter_prediction(self, current_frame, ref_frame, block_size=None, search_range=None):

    #     if block_size == None: block_size = self.block_size
    #     if search_range == None: search_range = self.search_range

    #     mvs = []
    #     total_mae = 0.0
        
    #     for y in range(0, current_frame.shape[0], block_size):
    #         for x in range(0, current_frame.shape[1], block_size):
    #             # if (x >= 0 and x+block_size <= ref_frame.shape[1] and
    #                 # y >= 0 and y+block_size <= ref_frame.shape[0]):
    #             # print("x,y: ", x,y)
    #             current_block = current_frame[y:y+block_size, x:x+block_size]
    #             mv, mae = self.find_best_match(current_block, ref_frame, x, y, block_size, search_range)
    #             # print("mv, mae", mv, mae)
    #             mvs.append(mv)
    #             total_mae += mae
                
    #     average_mae = total_mae / (len(mvs) or 1)
    #     return mvs, average_mae

    def visualize_comparison(self, img1, img2=None, img3=None, block_size=2, factor=1):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img1*factor, cmap="gray", vmin=0, vmax=255)
        # plt.title(f"{title[0].replace('_', ' ')}_block_{block_size}")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(img2*factor, cmap="gray", vmin=0, vmax=255)
        # # plt.title(f"{title[1].replace('_', ' ')}_block_{block_size}")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(img3*factor, cmap="gray", vmin=0, vmax=255)
        # # plt.title(f"{title[2].replace('_', ' ')}_block_{block_size}")
        plt.axis("off")

        
        # plt.legend('image 1','image 2')
        # plt.legend(['i=2'], loc='upper right')    
        plt.tight_layout()
        plt.show()
    
    def visualize_reference_frames(self, frame, ref_indices, block_size):
        fig, ax = plt.subplots()
        cax = fig.add_axes([0.27, 0.8, 0.5, 0.05]) # Adds an Axes for the colorbar

        ax.imshow(frame, cmap='gray', aspect='equal')

        # Define discrete colors for each reference frame index
        unique_refs = np.unique(ref_indices)
        colors = plt.cm.get_cmap('viridis', self.nRefFrames).colors
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(-0.5, self.nRefFrames+0.5, 1), cmap.N)

        # Create a colored overlay for each block based on the reference frame index
        for y in range(0, frame.shape[0], block_size):
            for x in range(0, frame.shape[1], block_size):
                block_idx_y = y // block_size
                block_idx_x = x // block_size
                ref_idx = ref_indices[block_idx_y, block_idx_x]
                color = cmap(norm(ref_idx))
                rect = patches.Rectangle((x, y), block_size, block_size, linewidth=1, edgecolor='none', facecolor=color, alpha=0.4)
                ax.add_patch(rect)

        # Create the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=np.arange(len(unique_refs)))
        cbar.set_ticklabels(unique_refs)
        cbar.set_label('Reference Frame Index')

        plt.axis('off')
        plt.show()

    def visualize_motion_vectors(self, frame, motion_vectors, block_size):
        fig, ax = plt.subplots()
        ax.imshow(frame, cmap='gray', aspect='equal')
        
        # Calculate the number of blocks in each dimension
        num_blocks_y = frame.shape[0] // block_size
        num_blocks_x = frame.shape[1] // block_size

        # Iterate over the motion vectors and plot them
        for idx, (mv_x, mv_y) in enumerate(motion_vectors):
            block_y = (idx // num_blocks_x) * block_size
            block_x = (idx % num_blocks_x) * block_size
            
            # Plot the motion vector as an arrow
            ax.arrow(block_x, block_y, mv_x, mv_y, head_width=1, head_length=1, fc='r', ec='r')
            
        plt.axis('off')  # Turn off the axis
        plt.show()
# Example usage:
# frame = np.array(...)  # Your frame data here
# ref_indices = np.array(...)  # Your reference frame index data here
# unique_refs = np.unique(ref_indices)  # Unique reference frame indices
# visualize_reference_frames(frame, ref_indices, unique_refs)
    
    def frac_me_reference_frame(self, ref_frames, block_size):
        all_frames=[]
        for ar in np.copy(ref_frames):
            ar=np.array(ar)
            # ax=ar.T
            # print(ax)
            rows=[]
            cols=[]
            for row in ar:
                # row = np.array([1., 2., 3., 4., 6., 0.])
                avg_row = ((row + np.roll(row, -1))/2.0)
                combined_avg_row=(np.vstack([row, avg_row]).flatten('F')[:-1])
                rows.append(combined_avg_row)
            for row in np.array(rows).T:
                # row = np.array([1., 2., 3., 4., 6., 0.])
                avg_row = ((row + np.roll(row, -1))/2.0)
                combined_avg_row=(np.vstack([row, avg_row]).flatten('F')[:-1])
                # print(combined_avg_row)
                cols.append(np.ceil(combined_avg_row))
            ref_frame_=np.array(cols).T
            # print(ref_frame[::2,::2],'\n\n!!!!!!!!!!!!!!')
            # print(ref_frame[1::2,1::2],'\n\n!!!!!!!!!!!!!!')
            all_frames.append(ref_frame_)
            # print(np.array(cols).T)
        return all_frames
    
    
    def frac_me_find_best_match(self, current_block, ref_frames, x, y, block_size=None, search_range=None):
        if block_size is None: 
            block_size = self.block_size
        if search_range is None: 
            search_range = self.search_range
        
        best_mae = float('inf')
        best_mv = (0, 0, 0)  # Including the reference frame index.

        # Iterate over all reference frames.
        for ref_idx, ref_frame in enumerate(ref_frames):
            for dx in range(-search_range*block_size, search_range*block_size + 1,block_size):
                for dy in range(-search_range*block_size, search_range*block_size + 1,block_size):
                    # if ref_idx>=len(ref_frames):
                    #     break
                    # Check reference block within the current reference frame boundaries
                    if 0 <= x+dx < ref_frame.shape[1] - block_size and 0 <= y+dy < ref_frame.shape[0] - block_size:
                        ref_block = ref_frame[y+dy:y+dy+block_size, x+dx:x+dx+block_size]
                        mae = self.compute_mae(current_block, ref_block)

                        if mae < best_mae:
                            best_mae = mae
                            best_mv = (dx//block_size, dy//block_size, ref_idx)
                        # elif mae == best_mae:
                        #     if self.is_better_mv(best_mv, (dx//block_size, dy//block_size, ref_idx)):
                        #         best_mv = (dx, dy, ref_idx)
        return best_mv, best_mae
    
    def calculate_inter_frame_residual(self, x, y, mv, current_block, ref_frames, block_size=None):
        if block_size is None: 
            block_size = self.block_size

        mv_x, mv_y = mv[0], mv[1] # Motion vector components
        ref_frame = ref_frames[mv[2]] # Reference frame corresponding to the third component of mv
        
        # Calculate the coordinates for the predicted block
        pred_y = y + mv_y
        pred_x = x + mv_x
        
        # Ensure the coordinates are within the reference frame boundaries
        if 0 <= pred_x < ref_frame.shape[1] - block_size and 0 <= pred_y < ref_frame.shape[0] - block_size:

            #predicted_block = ref_frame[pred_y:pred_y + block_size, pred_x:pred_x + block_size]
            #if 0 <= pred_x+block_size*2 < ref_frame.shape[1] - block_size and 0 <= pred_y+block_size*2 < ref_frame.shape[0] - block_size:
            if self.FMEEnable:
                if 0 <= pred_x+block_size*2 < ref_frame.shape[1] - block_size and 0 <= pred_y+block_size*2 < ref_frame.shape[0] - block_size:
                    predicted_block = ref_frame[pred_y:pred_y + block_size*2:2, pred_x:pred_x + block_size*2:2]
                else:
                    predicted_block = np.ones((block_size, block_size)) * 128
            else :
                predicted_block = ref_frame[pred_y:pred_y + block_size, pred_x:pred_x + block_size]
            #else:
            #   predicted_block = ref_frame[pred_y:pred_y + block_size, pred_x:pred_x + block_size]
        else:
            # Handle the case where the block is outside the boundaries
            # This might involve padding or other strategies
            predicted_block = self.handle_boundary_conditions(ref_frame, pred_y, pred_x, block_size)
        
        residual = self.generate_residual_block(current_block, predicted_block)
        
        return residual

    def inter_prediction(self, current_frame, ref_frames, block_size=None, search_range=None, nRefFrames=None, fast_me=False, mvp=(0, 0)):
        if block_size is None: 
            block_size = self.block_size
        if search_range is None: 
            search_range = self.search_range
        if nRefFrames is None: 
            nRefFrames = self.nRefFrames

        mvs = []
        residual_per_block = []
        total_mae = 0.0
        
        sub_block_size = block_size//2
        #print("FRAME")
        for y in range(0, current_frame.shape[0], block_size):
            for x in range(0, current_frame.shape[1], block_size):

                # VBS metrics are initialized to None to reflect that they wont be populated unless VBSEnable is set
                vbs_mvs = None
                vbs_residuals = None
                vbs_mae = None

                if self.VBSEnable == True and x != 0 and y != 0: # If VBS is enabled, we process the sub blocks in parallel to the parent block
                    vbs_mvs = []
                    vbs_residuals = []
                    vbs_mae = 0
                    
                    for y_vbs in range(y, y+block_size, sub_block_size):
                        for x_vbs in range(x, x+block_size, sub_block_size):
                            current_block_vbs = current_frame[y_vbs:y_vbs+sub_block_size, x_vbs:x_vbs+sub_block_size]

                            if fast_me:
                                #mv, mae = self.fast_motion_estimation(current_block_vbs, ref_frames, x_vbs, y_vbs, sub_block_size, mvp, nRefFrames)
                                
                                if self.FMEEnable:
                                    mv, mae = self.fast_motion_estimation(current_block_vbs, ref_frames, x_vbs*2, y_vbs*2, sub_block_size, mvp, nRefFrames)
                                    residual = self.calculate_inter_frame_residual(x_vbs*2, y_vbs*2, mv, current_block_vbs, ref_frames, sub_block_size)
                                else :
                                    mv, mae = self.fast_motion_estimation(current_block_vbs, ref_frames, x_vbs, y_vbs, sub_block_size, mvp, nRefFrames)
                                    residual = self.calculate_inter_frame_residual(x_vbs, y_vbs, mv, current_block_vbs, ref_frames, sub_block_size)
                            else: 
                                #mv, mae = self.find_best_match(current_block_vbs, ref_frames, x_vbs, y_vbs, sub_block_size, search_range)
                                
                                if self.FMEEnable:
                                    mv, mae = self.find_best_match(current_block_vbs, ref_frames, x_vbs*2, y_vbs*2, sub_block_size, search_range)
                                    residual = self.calculate_inter_frame_residual(x_vbs*2, y_vbs*2, mv, current_block_vbs, ref_frames, sub_block_size)
                                else:
                                    mv, mae = self.find_best_match(current_block_vbs, ref_frames, x_vbs, y_vbs, sub_block_size, search_range)
                                    residual = self.calculate_inter_frame_residual(x_vbs, y_vbs, mv, current_block_vbs, ref_frames, sub_block_size)
                            
                            # Calculate residuals

                            vbs_mvs.append(mv)
                            vbs_residuals.append(residual)
                            vbs_mae = vbs_mae + mae

                    vbs_mae = vbs_mae/len(vbs_mvs)
                    
                # The encoding and processing for the complete block is enabled regardless of whether VBS is enabled  
                current_block = current_frame[y:y+block_size, x:x+block_size]

                if fast_me:
                    #yet to set fast_motion_estimation for FME
                    if self.FMEEnable:
                        mv, mae = self.fast_motion_estimation(current_block, ref_frames, x*2, y*2, block_size, mvp, nRefFrames)
                        residual = self.calculate_inter_frame_residual(2*x, 2*y, mv, current_block, ref_frames, block_size )
                    else :
                        mv, mae = self.fast_motion_estimation(current_block, ref_frames, x, y, block_size, mvp, nRefFrames)
                        residual = self.calculate_inter_frame_residual(x, y, mv, current_block, ref_frames, block_size )
                else:
                    if self.FMEEnable:
                        mv, mae = self.find_best_match(current_block, ref_frames, x*2, y*2, block_size, search_range)
                        residual = self.calculate_inter_frame_residual(2*x, 2*y, mv, current_block, ref_frames, block_size )
                    else:
                        mv, mae = self.find_best_match(current_block, ref_frames, x, y, block_size, search_range)
                        #print("X: ", x)
                        #print("Y: ", y)
                        #print("mae: ", mae)
                        #print("Ref Frames shape: ", ref_frames[0].shape)
                        residual = self.calculate_inter_frame_residual(x, y, mv, current_block, ref_frames, block_size )
                
                if self.VBSEnable and x != 0 and y != 0:
                    RD_cost_vbs = self.calculate_RD_cost(1, 1, vbs_mae, vbs_residuals, block_size, sub_block_size, self.lam)
                    RD_cost_bs  = self.calculate_RD_cost(1, 0, mae, residual, block_size, sub_block_size, self.lam)

                    if RD_cost_bs < RD_cost_vbs:
                        #print("RD_cost_bs: ", RD_cost_bs, " | RD_cost_vbs: ", RD_cost_vbs)
                        #print("Inter Selecting block size")
                        mvs.append(tuple((0, mv)))
                        residual_per_block.append(tuple((0, residual)))
                    else:
                        #print("RD_cost_bs: ", RD_cost_bs, " | RD_cost_vbs: ", RD_cost_vbs)
                        #print("Inter Selecting variable block size")
                        mvs.append(tuple((1, vbs_mvs)))
                        residual_per_block.append(tuple((1, vbs_residuals)))

                    mae = vbs_mae
                else:
                    mvs.append(tuple((0, mv)))
                    residual_per_block.append(tuple((0, residual)))
                
                total_mae += mae
                mvp = mv
                
        average_mae = total_mae / (len(mvs) or 1)
        return mvs, average_mae, residual_per_block


    #def find_best_match(self, current_block, ref_frames, x, y, block_size=None, search_range=None):
    #    if block_size is None: 
    #        block_size = self.block_size
    #    if search_range is None: 
    #        search_range = self.search_range
    #    
    #    best_mae = float('inf')
    #    best_mv = (0, 0, 0)  # Including the reference frame index.

    #    # Iterate over all reference frames.
    #    for ref_idx, ref_frame in enumerate(ref_frames):
    #        for dx in range(-search_range, search_range + 1):
    #            for dy in range(-search_range, search_range + 1):
    #                # if ref_idx>=len(ref_frames):
    #                #     break
    #                # Check reference block within the current reference frame boundaries
    #                if 0 <= x+dx < ref_frame.shape[1] - block_size and 0 <= y+dy < ref_frame.shape[0] - block_size:
    #                    ref_block = ref_frame[y+dy:y+dy+block_size, x+dx:x+dx+block_size]
    #                    mae = self.compute_mae(current_block, ref_block)
    
    #                    print("Best MAE: ", best_mae, " | MAE: ", mae)

    #                    if mae < best_mae:
    #                        best_mae = mae
    #                        best_mv = (dx, dy, ref_idx)
    #                    elif mae == best_mae:
    #                        if self.is_better_mv(best_mv, (dx, dy, ref_idx)):
    #                            best_mv = (dx, dy, ref_idx)
    #    return best_mv, best_mae

    def find_best_match(self, current_block, ref_frames, x, y, block_size=None, search_range=None):
        # print(type(ref_frames))
        # print(type(ref_frames[0]))
        if block_size is None: 
            block_size = self.block_size
        if search_range is None: 
            search_range = self.search_range
        
        best_mae = float('inf')
        best_mv = (0, 0, 0)  # Including the reference frame index.

        # Iterate over all reference frames.
        for ref_idx, ref_frame in enumerate(ref_frames):
            
            dx_range=search_range+1
            dy_range=search_range+1
            for dx in range(-search_range, dx_range ):
                for dy in range(-search_range, dy_range):
                    # if ref_idx>=len(ref_frames):
                    #     break
                    # Check reference block within the current reference frame boundaries
                    if 0 <= x+dx < ref_frame.shape[1] - block_size and 0 <= y+dy < ref_frame.shape[0] - block_size:
                        
                        #if 0 <= x+dx+block_size*2 < ref_frame.shape[1] - block_size and 0 <= y+dy+block_size*2 < ref_frame.shape[0] - block_size:
                        if self.FMEEnable:
                            if 0 <= x+dx+block_size*2 < ref_frame.shape[1] - block_size and 0 <= y+dy+block_size*2 < ref_frame.shape[0] - block_size:
                                ref_block = ref_frame[y+dy:y+dy+block_size*2:2, x+dx:x+dx+block_size*2:2]
                                mae = self.compute_mae(current_block, ref_block)
                                if mae < best_mae:
                                    best_mae = mae
                                    best_mv = (dx, dy, ref_idx)
                                elif mae == best_mae:
                                    if self.is_better_mv(best_mv, (dx, dy, ref_idx)):
                                        best_mv = (dx, dy, ref_idx)
                        else :
                            ref_block = ref_frame[y+dy:y+dy+block_size, x+dx:x+dx+block_size]
                            mae = self.compute_mae(current_block, ref_block)
                            if mae < best_mae:
                                best_mae = mae
                                best_mv = (dx, dy, ref_idx)
                            elif mae == best_mae:
                                if self.is_better_mv(best_mv, (dx, dy, ref_idx)):
                                    best_mv = (dx, dy, ref_idx)
                        # print(x+dx,y+dy)
                        
                        #print("Best MAE: ", best_mae, " | MAE: ", mae)

        return best_mv, best_mae
        
    def fast_motion_estimation(self, current_block, ref_frames, x, y, block_size, mvp, nRefFrames):
        best_mae = float('inf')
        best_mv = mvp
        best_ref_idx = 0
        
        for ref_idx, ref_frame in enumerate(ref_frames[:nRefFrames]):
            # Check all positions around the MVP
            for dx in range(mvp[0] - 1, mvp[0] + 2):
                for dy in range(mvp[1] - 1, mvp[1] + 2):
                    # print(type(ref_frame))
                    if 0 <= x + dx < ref_frame.shape[1] - block_size and 0 <= y + dy < ref_frame.shape[0] - block_size:

                        if 0 <= x+dx+block_size*2 < ref_frame.shape[1] - block_size and 0 <= y+dy+block_size*2 < ref_frame.shape[0] - block_size:
                            if self.FMEEnable:
                                ref_block = ref_frame[y+dy:y+dy+block_size*2:2, x+dx:x+dx+block_size*2:2]
                            else :
                                ref_block = ref_frame[y+dy:y+dy+block_size, x+dx:x+dx+block_size]

                            mae = self.compute_mae(current_block, ref_block)
                            if mae < best_mae:
                                best_mae = mae
                                best_mv = (dx, dy, ref_idx)
                                best_ref_idx = ref_idx

        return best_mv, best_ref_idx

    def get_search_points(self, mvp, ref_idx):
        # Define your search pattern here
        # Example pattern: [(0,0), (1,0), (0,1), (-1,0), (0,-1)]
        pattern = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]
        return [(mvp[0] + dx, mvp[1] + dy, ref_idx) for dx, dy in pattern]
    
    def handle_boundary_conditions(self, ref_frame, y, x, block_size):
        frame_height, frame_width = ref_frame.shape
        padded_block = np.zeros((block_size, block_size), dtype=ref_frame.dtype)

        # Calculate the overlap between the desired block and the actual frame
        y_start = max(y, 0)
        y_end = min(y + block_size, frame_height)
        x_start = max(x, 0)
        x_end = min(x + block_size, frame_width)

        # Determine the size of the overlap
        overlap_height = y_end - y_start
        overlap_width = x_end - x_start

        # Ensure there is an overlap to copy
        if overlap_height > 0 and overlap_width > 0:
            padded_block[y_start-y:y_start-y+overlap_height, x_start-x:x_start-x+overlap_width] = ref_frame[y_start:y_end, x_start:x_end]

        return padded_block



    # Helper method to compare motion vectors if MAE is the same.
    def is_better_mv(self, mv1, mv2):
        # Compare the L1 norm of the motion vectors and the reference frame index.
        return (abs(mv2[0]) + abs(mv2[1]), mv2[2]) < (abs(mv1[0]) + abs(mv1[1]), mv1[2])
    
    def generate_residual_block(self, current_block, predicted_block):
        return current_block - predicted_block
    
    # Apply 2D DCT to an input block.
    def apply_2d_dct(self, input_block):
        # Apply 2D DCT
        transformed_block = dct(dct(input_block, axis=0, norm='ortho'), axis=1, norm='ortho')
        # Round to the nearest integer
        transformed_block = np.round(transformed_block).astype(int)
        return transformed_block

    # Quantize the transform coefficients (TC) using the Q matrix.
    def quantize_TC(self, TC, Q):
        
        # try: QTC = np.round(TC / Q).astype(int)
        # except: print(TC,Q)
        QTC = np.round(TC / Q).astype(int)
        # except: print(TC,Q)
        return QTC

    # Round off the residual error to the nearest 2s power
    def approximate_residual_block(self, arr, n):
        # Handle negative values
        is_negative = arr < 0
        arr_abs = np.abs(arr)

        # Calculate the nearest power of 2 for positive, non-zer:470
        # o values
        powers_of_2 = np.where(arr_abs > 0, 2 ** np.round(np.log2(arr_abs)), 0)

        # Restore the sign for negative values
        powers_of_2 = np.where(is_negative, -powers_of_2, powers_of_2)

        # Convert the result to int32
        powers_of_2 = powers_of_2.astype(np.int32)

        return powers_of_2

    # Apply 2D Inverse DCT (IDCT) to the given block.
    def apply_2d_idct(self, transformed_coefficient_block):
        # Apply 2D IDCT
        residual_block = idct(idct(transformed_coefficient_block, axis=0, norm='ortho'), axis=1, norm='ortho')

        # Round the result to nearest integer for practical purposes
        residual_block = np.round(residual_block).astype(int)

        return residual_block

    # Rescale the quantized transform coefficients (QTC) using the Q matrix.
    def rescale_QTC(self, QTC, Q):
        return QTC * Q

    # Reconstruct the block by adding the approximated residual to the predicted block.
    def reconstruct_block(self, predicted_block, residual_block, Q):
        rescale_quant_trans = self.rescale_QTC(residual_block, Q)
        inv_transform_block = self.apply_2d_idct(rescale_quant_trans)
        return (predicted_block + inv_transform_block).astype(np.uint8)
        #return (predicted_block + residual_block).astype(np.uint8)

    # Reconstruct the entire frame using motion vectors, ref frame, and approximated residuals.
    def reconstruct_frame(self, mvs, ref_frames, approximated_residual_blocks, block_size,frame_no):
        # print('aaaaaaaaaaaaaaaaaaaaaaaaaaa')
        reconstructed_frame = np.zeros_like(ref_frames[0]).astype(np.uint8)
        if self.FMEEnable:
            ref_frames_fme=self.frac_me_reference_frame(ref_frames, block_size)
        # print('a')
        for idx, mv in enumerate(mvs):
            if idx%(self.w_pixels/self.block_size) == 0 and self.RCflag:
                self.changed_qp(self.qp_table[frame_no,idx//(self.w_pixels//self.block_size)])
            # print('aaaaaaaa')
            # print('aa')
            # if idx%(self.w_pixels/self.block_size) == 0 and self.RCflag:
            #     self.changed_qp(idx//(self.w_pixels/self.block_size))
            if mv[0] == 0: # No split
                block_y = (idx // (ref_frames[mv[1][2]].shape[1] // block_size)) * block_size
                block_x = (idx % (ref_frames[mv[1][2]].shape[1] // block_size)) * block_size

                #ref_frame = ref_frames[mv[1][2]]  # Reference frame corresponding to the third component of mv
                if self.FMEEnable:
                    ref_frame = ref_frames_fme[mv[1][2]]
                else:
                    ref_frame = ref_frames[mv[1][2]]  # Reference frame corresponding to the third component of mv
                mv_x, mv_y = mv[1][0], mv[1][1]  # Motion vector components

                # Calculate the coordinates for the predicted block
                if self.FMEEnable:
                    pred_y = 2*block_y + mv_y
                    pred_x = 2*block_x + mv_x
                else:
                    pred_y = block_y + mv_y
                    pred_x = block_x + mv_x

                # Ensure the coordinates are within the reference frame boundaries
                if 0 <= pred_x < ref_frame.shape[1] - block_size and 0 <= pred_y < ref_frame.shape[0] - block_size:
                    if self.FMEEnable:
                        if 0 <= pred_x+block_size*2 < ref_frame.shape[1] - block_size and 0 <= pred_y+block_size*2 < ref_frame.shape[0] - block_size:
                            predicted_block = ref_frame[pred_y:pred_y + block_size*2:2, pred_x:pred_x + block_size*2:2]
                        else:
                            predicted_block = np.ones((block_size, block_size)) * 128 

                        #predicted_block = ref_frame[pred_y:pred_y + block_size*2:2, pred_x:pred_x + block_size*2:2]
                    else: predicted_block = ref_frame[pred_y:pred_y + block_size, pred_x:pred_x + block_size]
                    #predicted_block = ref_frame[pred_y:pred_y + block_size, pred_x:pred_x + block_size]
                else:
                    # Handle the case where the block is outside the boundaries
                    # This might involve padding or other strategies
                    predicted_block = self.handle_boundary_conditions(ref_frame, pred_y, pred_x, block_size)
                # print(block_y, mv[1])
                # predicted_block = ref_frames[mv[2]][block_y + mv[1]:block_y + mv[1] + block_size, block_x + mv[0]:block_x + mv[0] + block_size]
                
                reconstructed_block = self.reconstruct_block(predicted_block, approximated_residual_blocks[idx][1], self.Q)
            else: # Split
                reconstructed_block = np.ones((block_size, block_size)) * 0
                for sb_mv_id, sb_mv in enumerate(mv[1]):
                    block_y = (idx // (ref_frames[sb_mv[2]].shape[1] // block_size)) * block_size
                    block_x = (idx % (ref_frames[sb_mv[2]].shape[1] // block_size)) * block_size
                    
                    vbs_block_y = block_y
                    vbs_block_x = block_x

                    if sb_mv_id == 1:
                        vbs_block_x = vbs_block_x + block_size//2
                    elif sb_mv_id == 2:
                        vbs_block_y = vbs_block_y + block_size//2
                    elif sb_mv_id == 3:
                        vbs_block_x = vbs_block_x + block_size//2
                        vbs_block_y = vbs_block_y + block_size//2

                    if self.FMEEnable:
                        ref_frame = ref_frames_fme[sb_mv[2]]
                    else:
                        ref_frame = ref_frames[sb_mv[2]]   # Reference frame corresponding to the third component of mv
                    mv_x, mv_y = sb_mv[0], sb_mv[1]  # Motion vector components

                    # Calculate the coordinates for the predicted block
                    if self.FMEEnable:
                        pred_y = 2*vbs_block_y + mv_y
                        pred_x = 2*vbs_block_x + mv_x
                    else:
                        pred_y = vbs_block_y + mv_y
                        pred_x = vbs_block_x + mv_x

                    # Ensure the coordinates are within the reference frame boundaries
                    if 0 <= pred_x < ref_frame.shape[1] - block_size//2 and 0 <= pred_y < ref_frame.shape[0] - block_size//2:
                        #predicted_block = ref_frame[pred_y:pred_y + block_size//2, pred_x:pred_x + block_size//2]
                        if self.FMEEnable:
                            if 0 <= pred_x+block_size < ref_frame.shape[1] - block_size and 0 <= pred_y+block_size < ref_frame.shape[0] - block_size:
                                predicted_block = ref_frame[pred_y:pred_y + block_size:2, pred_x:pred_x + block_size:2]
                            else:
                                predicted_block = np.ones((block_size//2, block_size//2)) * 128
                            #predicted_block = ref_frame[pred_y:pred_y + block_size:2, pred_x:pred_x + block_size:2]
                        else: predicted_block = ref_frame[pred_y:pred_y + block_size//2, pred_x:pred_x + block_size//2]
                    else:
                        # Handle the case where the block is outside the boundaries
                        # This might involve padding or other strategies
                        predicted_block = self.handle_boundary_conditions(ref_frame, pred_y, pred_x, block_size//2)
                    # print(block_y, mv[1])
                    # predicted_block = ref_frames[mv[2]][block_y + mv[1]:block_y + mv[1] + block_size, block_x + mv[0]:block_x + mv[0] + block_size]
                    if sb_mv_id == 0:
                        reconstructed_block[0:block_size//2, 0:block_size//2] = self.reconstruct_block(predicted_block, approximated_residual_blocks[idx][1][sb_mv_id], self.Qm1)
                    elif sb_mv_id == 1:
                        reconstructed_block[0:block_size//2, block_size//2:block_size] = self.reconstruct_block(predicted_block, approximated_residual_blocks[idx][1][sb_mv_id], self.Qm1)
                    elif sb_mv_id == 2:
                        reconstructed_block[block_size//2:block_size, 0:block_size//2] = self.reconstruct_block(predicted_block, approximated_residual_blocks[idx][1][sb_mv_id], self.Qm1)
                    elif sb_mv_id == 3:
                        reconstructed_block[block_size//2:block_size, block_size//2:block_size] = self.reconstruct_block(predicted_block, approximated_residual_blocks[idx][1][sb_mv_id], self.Qm1)

            reconstructed_frame[block_y:block_y+block_size, block_x:block_x+block_size] = reconstructed_block

        return reconstructed_frame

    def calculate_metrics(self, original_frame, modified_frame):
        return psnr(original_frame, modified_frame, data_range=255), ssim(original_frame, modified_frame, win_size=11, multichannel=False)

    #Generate the Q matrix based on the given 'i' and 'QP'.
    def generate_Q_matrix(self, i, QP):
        Q = np.zeros((i, i), dtype=int)
        for x in range(i):
            for y in range(i):
                if x + y < i - 1:     Q[x][y] = 2 ** QP
                elif x + y == i - 1:  Q[x][y] = 2 ** (QP + 1)
                else:                 Q[x][y] = 2 ** (QP + 2)
        # if(QP==0):
        #     print(i,QP,Q)
            # exit()
        return Q

    # Edit Qp
    def set_Qp (self, Qp):
        self.Qp = Qp
        self.Q = self.generate_Q_matrix(self.block_size, Qp)

    # Plot PSNR and SSIM
    def plot_psnr_ssim(self, block_sizes, psnr_values, ssim_values):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(block_sizes, psnr_values, marker='o')
        plt.title("Average PSNR vs Block Size")
        plt.xlabel("Block Size")
        plt.ylabel("Average PSNR")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(block_sizes, ssim_values, marker='o')
        plt.title("Average SSIM vs Block Size")
        plt.xlabel("Block Size")
        plt.ylabel("Average SSIM")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def dump_approximated_residual_to_file(self, approximated_residuals, width, height, block_size, filename="residual_values.txt"):
        with open(filename, 'w') as f:
            block_idx = 0
            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    residual_block = approximated_residuals[block_idx]
                    
                    # Writing the block's top-left coordinates before dumping its values
                    f.write(f"BLOCK ({x},{y}):\n")
                    
                    for row in residual_block:
                        f.write(','.join(map(str, row)) + "\n")
                    f.write("\n")  # Separate blocks by an empty line
                    block_idx += 1

    # Function to save Y-only data into a file
    def save_y_only(self, filename, y_data_list):
        with open(filename, 'wb') as f:
            for data in y_data_list:
                f.write(data.tobytes())

    # Retrieve encoded package
    def get_encoded_package(self):
        if self.encoded_package.any():
            return self.encoded_package
        
        return None

    # Find best match in the horizontal direction using left-i
    def intra_find_best_match_horizontal(self, current_block, ref_frame, x, y, block_size=None, search_range=None):
        
        if block_size == None: block_size = self.block_size
        if search_range == None: search_range = self.search_range

        best_mae = float('inf')
        best_mv = 0

        residuals = None

        if x == 0:
            best_mv = -1
            ref_block = np.ones((block_size, block_size)) * 128
            best_mae  = self.compute_mae(current_block, ref_block)
            residuals = current_block - ref_block

        else:
            for dx in range(-search_range, search_range + 1):
                # Ensure the reference block is entirely within the frame boundaries
                if (x+dx >= 0 and x+dx+block_size <= ref_frame.shape[1]):
                    ref_block = ref_frame[y:y+block_size, x+dx:x+dx+block_size]
                    mae = self.compute_mae(current_block, ref_block)

                    # Check if this block is a better match than the previous ones
                    if mae < best_mae:     
                        best_mae = mae
                        best_mv = dx
                        residuals = current_block - ref_block

                    # If MAE is the same, check motion vector magnitude
                    elif mae == best_mae: 
                        if abs(dx) < abs(best_mv) or abs(dx) == abs(best_mv):
                            best_mv = dx 
                            residuals = current_block - ref_block
                            # print(residuals)
                            # exit()

        return best_mv, best_mae, residuals

    # Find best match in the vertical direction using top-i
    def intra_find_best_match_vertical(self, current_block, ref_frame, x, y, block_size=None, search_range=None):
        
        if block_size == None: block_size = self.block_size
        if search_range == None: search_range = self.search_range

        best_mae = float('inf')
        best_mv = 0

        residuals = None

        if y == 0:
            best_mv = -1
            ref_block = np.ones((block_size, block_size)) * 128
            best_mae  = self.compute_mae(current_block, ref_block)
            residuals = current_block - ref_block

        else:
            for dy in range(-search_range, search_range + 1):
                # Ensure the reference block is entirely within the frame boundaries
                if (y+dy >= 0 and y+dy+block_size <= ref_frame.shape[0]):
                    ref_block = ref_frame[y+dy:y+dy+block_size, x:x+block_size]
                    mae = self.compute_mae(current_block, ref_block)

                    # Check if this block is a better match than the previous ones
                    if mae < best_mae:     
                        best_mae = mae
                        best_mv = dy
                        residuals = current_block - ref_block

                    # If MAE is the same, check motion vector magnitude
                    elif mae == best_mae: 
                        if abs(dy) < abs(best_mv) or abs(dy) == abs(best_mv):
                            best_mv = dy
                            residuals = current_block - ref_block

        return best_mv, best_mae, residuals

    # Entropy encoder for Residual blocks
    def entropy_encoder_block(self, residual_block,block_size):
        non_zero_count = 0
        non_zero_values = []
        result=[]
        n = block_size
        result = []
        flag=1
        zero_count=0

        for k in range(2 * n - 1):
            if k < n:
                i, j = 0, k
            else:
                i, j = k - n + 1, n - 1
            while i < n and j >= 0:
                # print(i,j)
                if residual_block[i][j]!=0:
                    if flag==0:
                        if zero_count:
                            result.append(zero_count)
                            zero_count=0
                        non_zero_values = []
                        non_zero_count=0
                        flag=1
                    non_zero_values.append(residual_block[i][j])
                    non_zero_count+=1
                    # flag=1
                else:
                    # print('else:',i,j)
                    if flag==1:
                        if non_zero_count:
                            result.append(-non_zero_count)
                            result.extend(non_zero_values)
                            non_zero_values=[]
                            non_zero_count=0
                        zero_count=0
                        flag=0
                    zero_count+=1
                i += 1
                j -= 1
        # print(i,j,'asdfa')
        if(non_zero_count):
            result.append(-non_zero_count)
            result.extend(non_zero_values)
        if zero_count:
            result.extend([0])
        
        return result
    
    def calculate_RD_cost(self, frame_type, split, mae, residuals, block_size=None, sub_block_size=None, lam=None):
        
        if block_size == None: block_size = self.block_size
        if sub_block_size == None: sub_block_size = self.sub_block_size
        if lam == None: lam = self.lam

        bit_rate = 0

        if split == 0: # calculate RD cost for the complete block
            entropy_encoded_residual = self.entropy_encoder_block(self.quantize_TC(self.apply_2d_dct(residuals), self.Q), block_size)
            if frame_type == 0: # Intra frame
                bit_rate = 8 # For intra frame we send only a single integer (8bits) value as motion vector
                #print("NO_SPLIT | BIT RATE cost: ", bit_rate, "; MAE: ", mae, "; RD_COST: ", lam*bit_rate+mae)
            else: # inter frame
                bit_rate = 8 * 2 # In inter frame we send 2 integers (8 bits) values as motion vector
                #print("NO_SPLIT | BIT RATE cost: ", bit_rate, "; MAE: ", mae, "; RD_COST: ", lam*bit_rate+mae)
            bit_rate = bit_rate + 8 * len(entropy_encoded_residual)
        else: # Calculate RD cost for the sub blocks
            if frame_type == 0: # Intra frame
                bit_rate = 8 * 4 # We will have to send 4 short ints since there are 4 sub blocks
                #print("SPLIT | BIT RATE cost: ", bit_rate, "; MAE: ", mae, "; RD_COST: ", lam*bit_rate+mae)
            else: # inter frame
                bit_rate = 8 * 4 * 2 # We will have to send 4 pairs of short ints since there are 4 sub blocks
                #print("SPLIT | BIT RATE cost: ", bit_rate, "; MAE: ", mae, "; RD_COST: ", lam*bit_rate+mae)
            
            for i in range(0, len(residuals)):
                entropy_encoded_residual = self.entropy_encoder_block(self.quantize_TC(self.apply_2d_dct(residuals[i]), self.Qm1), sub_block_size)
                bit_rate = bit_rate + 8 * len(entropy_encoded_residual)
        
        # RD_cost = lambda * Bitrate + Distrotion
        #print("Output: ", lam * bit_rate + mae)
        return lam * bit_rate + mae 

    def intra_prediction(self, current_frame, mode=0, block_size=None, search_range=None):
        
        if block_size == None: block_size = self.block_size
        if search_range == None: search_range = self.search_range

        mvs = []
        residual_per_block = []
        total_mae = 0.0

        ref_frame = np.ones((self.h_pixels, self.w_pixels)) * 128

        #print("FRAME")
        for y in range(0, current_frame.shape[0], block_size):
            for x in range(0, current_frame.shape[1], block_size):
                
                # VBS metrics are initialized to None to reflect that they wont be populated unless VBSEnable is set
                vbs_mvs = None
                vbs_residuals = None
                vbs_mae = None

                # If variable block sizes are enabled, then process the sub-blocks in parallel to the main block
                if self.VBSEnable == True and x != 0 and y != 0:
                    vbs_mvs = []
                    vbs_residuals = []
                    vbs_mae = 0

                    sub_block_size = block_size//2
                    
                    # Sub blocks processed in a "Z" pattern
                    for y_vbs in range(y, y+block_size, sub_block_size):
                        for x_vbs in range(x, x+block_size, sub_block_size):
                            current_block_vbs = current_frame[y_vbs:y_vbs+sub_block_size, x_vbs:x_vbs+sub_block_size]
                            
                            if mode==0:
                                mv, mae, residual = self.intra_find_best_match_horizontal(current_block_vbs, ref_frame, x_vbs, y_vbs, sub_block_size, search_range)
                            else:
                                mv, mae, residual = self.intra_find_best_match_vertical(current_block_vbs, ref_frame, x_vbs, y_vbs, sub_block_size, search_range)

                            vbs_mvs.append(mv)
                            vbs_residuals.append(residual)
                            #print("VBS MAE POINT: ", mae)
                            vbs_mae = vbs_mae + mae
                    
                    #print("VBS MAE POINT: ", vbs_mae)
                    vbs_mae = vbs_mae/len(vbs_mvs)
                    #print("VBS MAE Average: ", vbs_mae)

                # The encoding and processing for the complete block is enabled regardless of whether VBS is enabled  
                current_block = current_frame[y:y+block_size, x:x+block_size]
                    
                if mode == 0:
                    mv, mae, residual = self.intra_find_best_match_horizontal(current_block, ref_frame, x, y, block_size, search_range)
                else:
                    mv, mae, residual = self.intra_find_best_match_vertical(current_block, ref_frame, x, y, block_size, search_range)

                # If VBS is enabled we perform RD analysis to figure out whether to use VBS or standard block size
                if self.VBSEnable and x != 0 and y != 0:
                    RD_cost_vbs = self.calculate_RD_cost(0,1,vbs_mae, vbs_residuals, block_size, sub_block_size, self.lam)
                    RD_cost_bs  = self.calculate_RD_cost(0,0,mae,residual,block_size, sub_block_size, self.lam)

                    if RD_cost_bs < RD_cost_vbs:
                        #print("RD_cost_bs: ", RD_cost_bs, " | RD_cost_vbs: ", RD_cost_vbs)
                        #print("[",self.lam,"] Intra Selecting block size")
                        mvs.append(tuple((0, mv)))
                        residual_per_block.append(tuple((0, residual)))
                    else:
                        #print("RD_cost_bs: ", RD_cost_bs, " | RD_cost_vbs: ", RD_cost_vbs)
                        #print("[", self.lam, "] Intra Selecting variable block size")
                        mvs.append(tuple((1, vbs_mvs)))
                        residual_per_block.append(tuple((1, vbs_residuals)))

                    mae = vbs_mae
                else:
                    mvs.append(tuple((0, mv)))
                    residual_per_block.append(tuple((0, residual)))

                if x == 0 and mode == 0:
                    reconstructed_block = np.ones((block_size, block_size)) * 128 + residual
                elif y == 0 and mode == 1:
                    reconstructed_block = np.ones((block_size, block_size)) * 128 + residual
                elif mode == 0:
                    reconstructed_block = ref_frame[y:y+block_size, (x+mv):(x+mv)+block_size] + residual
                elif mode == 1:
                    reconstructed_block = ref_frame[(y+mv):(y+mv)+block_size, x:x+block_size] + residual

                ref_frame[y:y+block_size, x:x+block_size] = reconstructed_block
                #if self.VBSEnable:
                #    i = 0
                #    sub_block_size = block_size//2
                #    for y_vbs in range(y, y+block_size, sub_block_size):
                #        for x_vbs in range(x, x+block_size, sub_block_size):
                #            residual_frame[y_vbs:y_vbs+sub_block_size, x_vbs:x_vbs+sub_block_size] = vbs_residuals[i]
                #            i = i + 1
                #else:                            
                #    residual_frame[y:y+block_size, x:x+block_size] = residual

                total_mae += mae

        average_mae = total_mae / (len(mvs) or 1)

        return mvs, average_mae, residual_per_block, ref_frame
    
    # Reconstruct Frame for intra prediction
    def reconstruct_frame_intra(self, mode, mvs, approximated_residual_blocks_per_frame, block_size,frame_no):
        reconstructed_frame      = np.ones((self.h_pixels, self.w_pixels)) * 128
        residual_frame           = np.ones((self.h_pixels, self.w_pixels)) * 0

        index = 0

        rescale_inv_trans_residuals = []

        for idx,block in enumerate(approximated_residual_blocks_per_frame):
            row_no=idx//(self.w_pixels//self.block_size)
            if idx%(self.w_pixels/self.block_size) == 0 and self.RCflag:
                    self.changed_qp(self.qp_table[frame_no,row_no])
                # print('aaaaaaaa')
            if block[0] == 0: # No Split
                rescaled_block = self.rescale_QTC(block[1], self.Q)
                inv_transformed_block = self.apply_2d_idct(rescaled_block)
                rescale_inv_trans_residuals.append(tuple((0, inv_transformed_block)))
            else: # Split
                res_inv_trns_res_sub_blocks = []

                for subblocks in block[1]:
                    rescaled_block = self.rescale_QTC(subblocks, self.Qm1)
                    inv_transform_block = self.apply_2d_idct(rescaled_block)
                    res_inv_trns_res_sub_blocks.append(inv_transform_block)
                
                rescale_inv_trans_residuals.append(tuple((1, res_inv_trns_res_sub_blocks)))

        #rescale_quant_trans = [self.rescale_QTC(block, self.Q) for block in approximated_residual_blocks_per_frame]
        #inv_transform_block = [self.apply_2d_idct(block)       for block in rescale_quant_trans]
        #inv_transform_block = approximated_residual_blocks_per_frame

        #f = open("dummy_file_dump.txt", "a")

        for y in range(0, self.h_pixels, block_size):
            for x in range(0, self.w_pixels, block_size):
                if x == 0 and mode == 0:
                    reconstructed_block = np.ones((block_size, block_size)) * 128 + rescale_inv_trans_residuals[index][1]
                    residual_frame[y:y+block_size, x:x+block_size] = rescale_inv_trans_residuals[index][1]
                elif y == 0 and mode == 1:
                    reconstructed_block = np.ones((block_size, block_size)) * 128 + rescale_inv_trans_residuals[index][1]
                    residual_frame[y:y+block_size, x:x+block_size] = rescale_inv_trans_residuals[index][1]
                elif mode == 0:
                    if rescale_inv_trans_residuals[index][0] == 0: # No Split
                        reconstructed_block = reconstructed_frame[y:y+block_size, (x+mvs[index][1]):(x+mvs[index][1])+block_size] + rescale_inv_trans_residuals[index][1]
                        residual_frame[y:y+block_size, x:x+block_size] = rescale_inv_trans_residuals[index][1]
                    else: # Split
                        j = 0
                        reconstructed_block = np.ones((block_size, block_size))
                        for y_vbs in range(y, y+block_size, block_size//2):
                            for x_vbs in range(x, x+block_size, block_size//2):
                                reconstructed_block[y_vbs-y:y_vbs-y+(block_size//2), x_vbs-x:x_vbs-x+(block_size//2)] = reconstructed_frame[y_vbs:y_vbs+(block_size//2), (x_vbs+mvs[index][1][j]):(x_vbs+mvs[index][1][j] + (block_size//2))] + rescale_inv_trans_residuals[index][1][j]
                                residual_frame[y_vbs:y_vbs+block_size//2, x_vbs:x_vbs+block_size//2] = rescale_inv_trans_residuals[index][1][j]
                                j = j + 1
                elif mode == 1:
                    if rescale_inv_trans_residuals[index][0] == 1: # No Split
                        reconstructed_block = reconstructed_frame[(y+mvs[index][1]):(y+mvs[index][1])+block_size, x:x+block_size] + rescale_inv_trans_residuals[index][1]
                        residual_frame[y:y+block_size, x:x+block_size] = rescale_inv_trans_residuals[index][1]
                    else: # Split
                        j = 0
                        reconstructed_block = np.ones((block_size, block_size))
                        for y_vbs in range(y, y+block_size, block_size//2):
                            for x_vbs in range(x, x+block_size, block_size//2):
                                reconstructed_block[(y_vbs-y):(y_vbs-y)+(block_size//2), (x_vbs-x):(x_vbs-x)+(block_size//2)] = reconstructed_frame[(y_vbs+mvs[index][1][j]):(y_vbs+mvs[index][1][j])+block_size//2, x_vbs:x_vbs+block_size//2] + rescale_inv_trans_residuals[index][1][j]
                                residual_frame[y_vbs:y_vbs+block_size//2, x_vbs:x_vbs+block_size//2] = rescale_inv_trans_residuals[index][1][j]
                                j = j + 1

                #f.write("\nReconstructed block: ")
                #f.write(str(reconstructed_block))
                reconstructed_frame[y:y+block_size, x:x+block_size] = reconstructed_block
                #residual_frame[y:y+block_size, x:x+block_size] = approximated_residual_blocks_per_frame[index]

                index = index + 1

        #f.close()

        reconstructed_frame = reconstructed_frame.astype(np.uint8)
        residual_frame      = residual_frame

        return reconstructed_frame, residual_frame

    # Differntial encoder for MV
    # def differential_encoder_frame(self, frame_type, mv_for_frame):
        
    #     frame_mvs    =  ""
    #     ref_intra_mv =  0
    #     ref_inter_mv = (0, 0)
        
    #     if frame_type == 0: # Intra frame
    #         for j in range(0, len(mv_for_frame)):
    #             mv = -(ref_intra_mv - mv_for_frame[j])
    #             if j==0: frame_mvs = frame_mvs + str(mv)
    #             else: frame_mvs = frame_mvs + ";" + str(mv)
    #             ref_intra_mv = mv_for_frame[j]
    #     else:                     # Inter frame
    #         for j in range(0, len(mv_for_frame)):
    #             mv = (-(ref_inter_mv[0] - mv_for_frame[j][0]), -(ref_inter_mv[1] - mv_for_frame[j][1]))
    #             if j==0: frame_mvs = frame_mvs + str(mv)
    #             else: frame_mvs = frame_mvs + ";" + str(mv)
    #             ref_inter_mv = mv_for_frame[j]
        
    #     return frame_mvs

    def differential_encoder_frame(self, frame_type, mv_for_frame):
        
        frame_mvs = ""
        ref_intra_mv = 0
        ref_inter_mv = (0, 0, 0)  # Including reference frame index

        if frame_type == 0: # Intra frame
            for j, mv in enumerate(mv_for_frame):

                if mv[0] == 0: # No split
                    diff_mv = mv[1] - ref_intra_mv
                    if j == 0:
                        frame_mvs = "0\'(" + str(diff_mv) + ")"
                    else:
                        frame_mvs += ";0\'(" + str(diff_mv) + ")"
                    
                    ref_intra_mv = mv[1]

                elif mv[0] == 1: #Split
                    for k, sb_mv in enumerate(mv[1]): # loop through the subblock MV
                        diff_mv = sb_mv - ref_intra_mv
                        if k == 0:
                            frame_mvs += ";1\'(" + str(diff_mv) + ","
                        elif k == 3:
                            frame_mvs += str(diff_mv) + ")"
                        else:
                            frame_mvs +=  str(diff_mv) + ","
                        
                        ref_intra_mv = sb_mv

        else: # Inter frame
            for j, mv in enumerate(mv_for_frame):
                # Calculate differential for x, y and reference frame index
                if mv[0] == 0: # No split
                    diff_mv = (mv[1][0] - ref_inter_mv[0], mv[1][1] - ref_inter_mv[1], mv[1][2] - ref_inter_mv[2])
                    if j == 0:
                        frame_mvs = "0\'" + str(diff_mv)
                    else:
                        frame_mvs += ";0\'" + str(diff_mv)
                    ref_inter_mv = mv[1]
                elif mv[0] == 1: # Split
                    for k, sb_mv in enumerate(mv[1]):
                        diff_mv = (sb_mv[0] - ref_inter_mv[0], sb_mv[1] - ref_inter_mv[1], sb_mv[2] - ref_inter_mv[2])
                        if k == 0:
                            frame_mvs += ";1\'(" + str(diff_mv) + ","
                        elif k == 3:
                            frame_mvs += str(diff_mv) + ")"
                        else:
                            frame_mvs += str(diff_mv) + ","
                        
                        ref_inter_mv = sb_mv
            
        return frame_mvs


    def entropy_encoder_frame(self, frame_residuals, block_size=None):
        if block_size == None: block_size = self.block_size

        residual_for_frame = ""
        
        for i, residual in enumerate(frame_residuals):
            if residual[0] == 0: # No split
                if i == 0:
                    residual_for_frame = "0\'(" + str(self.entropy_encoder_block(residual[1], block_size)) + ")"
                else:
                    residual_for_frame += ";0\'(" + str(self.entropy_encoder_block(residual[1], block_size)) + ")"
            elif residual[0] == 1: # Split
                for k, sb_residual in enumerate(residual[1]): # Loop through the residuals of the subblocks 
                    if k == 0:
                        residual_for_frame += ";1\'(" + str(self.entropy_encoder_block(sb_residual, block_size//2)) + ","
                    elif k == 3:
                        residual_for_frame += str(self.entropy_encoder_block(sb_residual, block_size//2)) + ")"
                    else:
                        residual_for_frame += str(self.entropy_encoder_block(sb_residual, block_size//2)) + ","

        return residual_for_frame

    def transmit_bitstream(self, intra_dur=None, block_size=None, mv_file=None, residual_file=None):
        
        if intra_dur == None: intra_dur = self.intra_dur
        if block_size == None: block_size = self.block_size

        if not self.encoded_package_f:
            print("[ERROR] No encoded package available, please run encode() first")
            return

        mvs_per_frame      = self.encoded_package["MVS per Frame"]
        residual_per_frame = self.encoded_package["approx residual"]

        f_trns_mvs_per_frame     = open(mv_file, "w")
        f_trns_mvs_per_frame_raw = open(f"files/mvs_per_frame_raw_Qp_{self.Qp}_I_{self.intra_dur}.txt", "w")

        f_trns_res_per_frame     = open(residual_file, "w")
        for i in range(0, len(mvs_per_frame)):
            mvs = mvs_per_frame[i]
            residuals = residual_per_frame[i]
            
            if i%intra_dur == 0: frame_type = 0
            else: frame_type = 1

            f_trns_mvs_per_frame.write(str(frame_type) + "|" + self.differential_encoder_frame(frame_type, mvs) + "\n")
            f_trns_mvs_per_frame_raw.write(str(frame_type) + "|" + str(mvs) + "\n")
            f_trns_res_per_frame.write(str(self.entropy_encoder_frame(residuals,block_size)) + "\n")
        
        f_trns_mvs_per_frame.close()
        f_trns_mvs_per_frame_raw.close()
        f_trns_res_per_frame.close()

    def encode(self, intra_mode=None, intra_dur=None, search_range=None, block_size=None, save_enc_pkg=True):
        encoded_package = {}

        mae_per_frame  = []
        mvs_per_frame  = []
        approximated_residual_blocks_per_frame = []
        # ref_frame      = np.ones((self.h_pixels, self.w_pixels)) * 128  # For the first frame
        ref_frames = [np.ones((self.h_pixels, self.w_pixels)) * 128]  # Start with one reference frame filled with 128
        reconstructed_frames = []
        frame_no       =[]
        psnr_per_frame = []
        ssim_per_frame = []

        if search_range == None: search_range = self.search_range
        if block_size   == None: block_size   = self.block_size
        if intra_dur    == None: intra_dur    = self.intra_dur
        if intra_mode   == None: intra_mode   = self.intra_mode

        for i in range(self.frames):
            temp_frame_budget=self.frame_budget
            
            nref_frames=[]
            mvs_frame=[]
            frame_no.append(i)
            current_padded_frame = self.pad_hw(self.y_only_f_arr[i], block_size, 128)
            # ref_padded_frame     = self.pad_hw(ref_frame, block_size, 128)
            quantized_blocks                    = []
            if i%intra_dur == 0:   # Intra
                #print(self.row_budget)
                last_row_bitsize = 0
                
                
                
                mvs, average_mae, intra_residual, _ = self.intra_prediction(current_padded_frame, intra_mode, block_size, search_range)            # Intra prediction
                for block_no,block in enumerate(intra_residual): # block is a tuple : (split, residual/s)
                    # block_no+=1
                    
                    if (block_no)%(self.w_pixels/self.block_size) == 0 and self.RCflag:
                        # print('reaching')
                        # exit()
                        # print(self.frame_budget)
                        # exit()
                        row_no=block_no//(self.w_pixels//self.block_size)
                        temp_frame_budget-=last_row_bitsize
                        # print('last row:',last_row_bitsize,'budget remain:',temp_frame_budget)
                        temp_row_budget = (temp_frame_budget)/(self.h_pixels/self.block_size-row_no)
                        # print(temp_row_budget)                      
                        temp_qp=self.get_qp_val(self.cif_qp_i_table,temp_row_budget)
                        # print(temp_qp)
                        self.changed_qp(temp_qp)
                        last_row_bitsize = 0
                        self.qp_table[i,row_no]=self.Qp
                        # print(self.Qp)
                        # print(block_no, temp_row_budget, last_row_bitsize,self.Qp,self.Qpm1)  
                        # exit()
                        # print(row_no,self.Qp,temp_row_budget)


                    if block[0] == 0: # No split
                        # if block_no==1:
                        #     temp_budget = self.row_budget
                        # else: 
                            

                        
                        transformed_block = self.apply_2d_dct(block[1])
                        quantized_block   = self.quantize_TC(transformed_block, self.Q)
                        quantized_blocks.append(tuple((0, quantized_block)))
                    
                    else: # Split
                        vbs_blocks = []
                        for sub_block in block[1]:
                            transformed_block = self.apply_2d_dct(sub_block)
                            quantized_block   = self.quantize_TC(transformed_block, self.Qm1)
                            vbs_blocks.append(quantized_block)
                    
                        quantized_blocks.append(tuple((1, vbs_blocks)))
                    # print(block_no,(quantized_blocks[-1][1]))
                    # print(len((((quantized_blocks[-1])[1]).tostring()).encode('utf-8')))

                    size_in_bits=0
                    if quantized_blocks[-1][0] == 1:
                        for bl in quantized_blocks[-1][1]:
                            x=self.entropy_encoder_block(bl,self.sub_block_size)
                            size_in_bits+=(len(x)+5)
                    else:
                        x=self.entropy_encoder_block(quantized_blocks[-1][1],self.block_size)
                        size_in_bits=len(x)+5
                    last_row_bitsize+=(size_in_bits)

                reconstructed_frame, residual_frame = self.reconstruct_frame_intra(intra_mode, mvs, quantized_blocks, block_size,i)
                #plt.imshow(reconstructed_frame, cmap="gray", vmin=0, vmax=255)
                #plt.show()
                #plt.imshow(residual_frame, cmap="gray", vmin=0, vmax=255)
                #plt.show()
                ref_frames = []
                for mv in (mvs):
                    if intra_mode==0:
                        mvs_new = (mv, 0)
                        mvs_frame.append(mvs_new)
                    else:
                        mvs_new = (0, mv)
                        mvs_frame.append(mvs_new)
                #approximated_residual_blocks_per_frame.append(quantized_blocks)
                # self.visualize_motion_vectors(reconstructed_frame, mvs_frame, block_size)
            
            else:                  # Inter
                quantized_blocks                    = []
                last_row_bitsize = 0
                if self.FMEEnable:
                    mvs, average_mae, inter_residual    = self.inter_prediction(current_padded_frame, self.frac_me_reference_frame(ref_frames, block_size), block_size, search_range*2, fast_me=self.fast_me)      # Inter prediction
                else :
                    mvs, average_mae, inter_residual    = self.inter_prediction(current_padded_frame, ref_frames, block_size, search_range, fast_me=self.fast_me)      # Inter prediction
                
                #mvs, average_mae, inter_residual        = self.inter_prediction(current_padded_frame, ref_frames, block_size, search_range, fast_me=self.fast_me)      # Inter prediction
                
                if self.FMEEnable:
                    ref_frames_fme=self.frac_me_reference_frame(ref_frames, block_size)
                
                for block_no,block in enumerate(inter_residual): # block is a tuple : (split, residual/s)
                    if (block_no)%(self.w_pixels/self.block_size) == 0 and self.RCflag:
                        row_no=block_no//(self.w_pixels//self.block_size)
                        temp_frame_budget-=last_row_bitsize
                        # print('last row:',last_row_bitsize,'budget remain:',temp_frame_budget)
                        temp_row_budget = (temp_frame_budget)/(self.h_pixels/self.block_size-row_no)
                        # print(temp_row_budget)                      
                        temp_qp=self.get_qp_val(self.cif_qp_p_table,temp_row_budget)
                        # print(temp_qp)
                        self.changed_qp(temp_qp)
                        last_row_bitsize = 0
                        self.qp_table[i,row_no]=self.Qp
                    if block[0] == 0:
                        transformed_block               = self.apply_2d_dct(block[1])
                        quantized_transform             = self.quantize_TC(transformed_block, self.Q)
                        quantized_blocks.append(tuple((0,quantized_transform)))
                    else:
                        vbs_blocks = [] 

                        for sub_block in block[1]:
                            transformed_block = self.apply_2d_dct(sub_block)
                            quantized_transform = self.quantize_TC(transformed_block, self.Qm1)
                            vbs_blocks.append(quantized_transform)

                         
                        #print("Encoded residuals all sub blocks:", vbs_blocks)
                        quantized_blocks.append(tuple((1, vbs_blocks)))

                    size_in_bits=0
                    if quantized_blocks[-1][0] == 1:
                        for bl in quantized_blocks[-1][1]:
                            x=self.entropy_encoder_block(bl,self.sub_block_size)
                            size_in_bits+=(len(x)+5)
                    else:
                        x=self.entropy_encoder_block(quantized_blocks[-1][1],self.block_size)
                        size_in_bits=len(x)+5
                    last_row_bitsize+=(size_in_bits)
                 
                #approximated_residual_blocks_per_frame.append(quantized_blocks)
                reconstructed_frame = self.reconstruct_frame(mvs, ref_frames, quantized_blocks, block_size,i)
                #plt.imshow(reconstructed_frame, cmap="gray", vmin=0, vmax=255)
                #plt.show()
                # if self.FMEEnable:
                #     reconstructed_frame = self.reconstruct_frame(fmvs, ref_frames, quantized_blocks, block_size)
                # self.visualize_comparison(current_padded_frame, ref_frames[0], reconstructed_frame)
                # for mv in (mvs):
                #     nref_frame = mv[2]
                #     nref_frames.append(nref_frame)
                #     mvs_new = (mv[0], mv[1])
                #     # print(mvs_new)
                #     mvs_frame.append(mvs_new)
                #     # print(mvs_frame)
                # self.visualize_reference_frames(reconstructed_frame, np.array(nref_frames).reshape(36,44), block_size)
                # self.visualize_motion_vectors(reconstructed_frame, mvs_frame, block_size)
            mvs_per_frame.append(mvs)
            mae_per_frame.append(average_mae)
            reconstructed_frames.append(reconstructed_frame)
            approximated_residual_blocks_per_frame.append(quantized_blocks)

            if i < self.frames - 1:  # Set the current reconstructed frame as the reference for the next one
                if len(ref_frames) >= self.nRefFrames:
                    ref_frames.pop(0)  # Remove the oldest reference frame if we've reached the limit
                ref_frames.append(reconstructed_frame)

            avg_psrn, avg_ssim = self.calculate_metrics(self.y_only_f_arr[i], reconstructed_frame)
            psnr_per_frame.append(avg_psrn)
            ssim_per_frame.append(avg_ssim)
        # print(mvs_per_frame)
        decoded_frames = self.decoder.decode(approximated_residual_blocks_per_frame, mvs_per_frame, intra_mode, intra_dur, block_size, self.frames, self.w_pixels, self.h_pixels)

        # Saving the decoded frames to a Y-only file
        self.save_y_only(f"yuv/y_only_decoded__Qp_{self.Qp}_I_{self.intra_dur}.yuv", decoded_frames)
        # print(mvs_per_frame)
        # Collect all calculated results, useful for debugging and printing
        encoded_package["block size"]       = block_size
        encoded_package["num frames"]       = self.frames
        encoded_package["height in pixels"] = self.h_pixels
        encoded_package["width in pixels"]  = self.w_pixels
        encoded_package["search range"]     = search_range
        encoded_package["PSNR per frame"]   = psnr_per_frame
        encoded_package["SSIM per frame"]   = ssim_per_frame
        encoded_package["MAE per Frame"]    = mae_per_frame
        encoded_package["MVS per Frame"]    = mvs_per_frame
        encoded_package["approx residual"]  = approximated_residual_blocks_per_frame
        self.encoded_package_f              = True

        if save_enc_pkg == True:
            self.encoded_package = encoded_package

        self.save_y_only(f"yuv/y_only_reconstructed_Qp_{self.Qp}_I_{self.intra_dur}.yuv", reconstructed_frames)

        # Comparing the decoded frames with the reconstructed frames
        # print(self.qp_table)
        # for i in range(self.frames):
        #     assert np.array_equal(decoded_frames[i], reconstructed_frames[i]), f"Frame {i} mismatch!"
        # print(self.qp_table)
        return psnr_per_frame
