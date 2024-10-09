import math
import time
import signal
import decoder
import numpy as np
from math import ceil
import multiprocessing as mp
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.fftpack import dct, idct
import matplotlib.patches as patches
from skimage.metrics import structural_similarity as ssim
from matplotlib.colors import ListedColormap, BoundaryNorm
from skimage.metrics import peak_signal_noise_ratio as psnr
# import parallelTestModule
# global_ref_buffer=[[[128]*352 for i in range(288)]]

# Y Only Video Codec
class Y_Video_codec:
    


    def init_worker(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def __init__(self, h_pixels, w_pixels, frames, block_size, search_range, Qp, intra_dur, intra_mode, lam=None, VBSEnable=False, nRefFrames=1, yuv_file=None,  y_only_frame_arr=None, fast_me=False, FMEEnable=False, RCFlag=None, targetBR=None, frame_rate=30, qp_rate_tables=None, intra_thresh=None, ParallelMode=0):
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
        self.num_blocks_per_row= w_pixels/block_size
        self.sub_block_size    = block_size//2
        self.search_range      = search_range
        self.Qp                = Qp
        self.const_init_Qp     = Qp
        self.Qpm1              = None
        self.intra_dur         = intra_dur
        self.intra_mode        = intra_mode
        self.Q                 = self.generate_Q_matrix(block_size, Qp)
        self.Qm1               = None
        self.decoder           = decoder.decoder(intra_mode, intra_dur, block_size, frames, h_pixels, w_pixels, Qp, nRefFrames, FMEEnable, lam, VBSEnable, False, RCFlag, targetBR, frame_rate, qp_rate_tables,ParallelMode=ParallelMode)
        self.encoded_package   = None
        self.encoded_package_f = False
        self.nRefFrames        = nRefFrames
        self.fast_me           = fast_me
        self.FMEEnable         = FMEEnable
        self.VBSEnable         = VBSEnable
        self.lam               = lam
        self.RCFlag            = RCFlag
        self.target_bitrate    = None
        self.bitrate_per_row   = None
        self.frame_rate        = frame_rate
        self.qr_rate_tables    = qp_rate_tables
        self.intra_thresh      = intra_thresh
        self.ParallelMode      = ParallelMode
        self.inter0 = []
        self.intra0 = []
        self.inter1 = []
        self.intra1 = []
        self.inter2 = []
        self.intra2 = []
        self.inter3 = []
        self.intra3 = []

        if Qp > 0:
            self.Qpm1 = self.Qp-1
            self.Qm1  = self.generate_Q_matrix(self.sub_block_size, self.Qpm1)
        else:
            self.Qpm1 = self.Qp
            self.Qm1  = self.generate_Q_matrix(self.sub_block_size, self.Qpm1)

        if targetBR != None:
            tokens = targetBR.split(" ")
            num    = int(tokens[0])
            
            if tokens[1] == "kbps":
                self.target_bitrate = num * 1024
            elif tokens[1] == "mbps":
                self.target_bitrate = num * 1048576
            else: # bps
                self.target_bitrate = num
            self.bitrate_per_row = (self.target_bitrate//self.frame_rate)/(self.h_pixels/self.block_size)

        if yuv_file != None:
            self.y_only_f_arr = self.read_yuv(yuv_file, h_pixels, w_pixels, frames)
        else:
            self.y_only_f_arr = y_only_frame_arr # numpy arr with index (frame, px height, px width)

    # Set target bitrate for Rate control
    def set_target_bitrate(self, targetBR):
        tokens = targetBR.split(" ")
        num    = int(tokens[0])
        
        if tokens[1] == "kbps":
            self.target_bitrate = num * 1024
        elif tokens[1] == "mbps":
            self.target_bitrate = num * 1048576
        else: # bps
            self.target_bitrate = num
            
        self.bitrate_per_row = (num//self.frame_rate)/(self.h_pixels/self.block_size)

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

      
    # Motion Estimation functions
    #Compute Mean Absolute Error between two blocks.
    def compute_mae(self, block1, block2):
        return np.mean(np.abs(block1 - block2))

    def visualize_comparison(self, img1, img2=None, img3=None, block_size=2, factor=1):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img1*factor, cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(img2*factor, cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(img3*factor, cmap="gray", vmin=0, vmax=255)
        plt.axis("off")    
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

'''Example usage:
frame = np.array(...)  # Your frame data here
ref_indices = np.array(...)  # Your reference frame index data here
unique_refs = np.unique(ref_indices)  # Unique reference frame indices
visualize_reference_frames(frame, ref_indices, unique_refs)'''
    
    def frac_me_reference_frame(self, ref_frames, block_size):
        all_frames=[]
        for ar in np.copy(ref_frames):
            ar=np.array(ar)
            # ax=ar.T
            # print(ax)
            rows=[]
            cols=[]
            for row in ar:
                avg_row = ((row + np.roll(row, -1))/2.0)
                combined_avg_row=(np.vstack([row, avg_row]).flatten('F')[:-1])
                rows.append(combined_avg_row)
            for row in np.array(rows).T:
                avg_row = ((row + np.roll(row, -1))/2.0)
                combined_avg_row=(np.vstack([row, avg_row]).flatten('F')[:-1])
                cols.append(np.ceil(combined_avg_row))
            ref_frame_=np.array(cols).T
            all_frames.append(ref_frame_)
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
                    # Check reference block within the current reference frame boundaries
                    if 0 <= x+dx < ref_frame.shape[1] - block_size and 0 <= y+dy < ref_frame.shape[0] - block_size:
                        ref_block = ref_frame[y+dy:y+dy+block_size, x+dx:x+dx+block_size]
                        mae = self.compute_mae(current_block, ref_block)

                        if mae < best_mae:
                            best_mae = mae
                            best_mv = (dx//block_size, dy//block_size, ref_idx)
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

            if self.FMEEnable:
                if 0 <= pred_x+block_size*2 < ref_frame.shape[1] - block_size and 0 <= pred_y+block_size*2 < ref_frame.shape[0] - block_size:
                    predicted_block = ref_frame[pred_y:pred_y + block_size*2:2, pred_x:pred_x + block_size*2:2]
                else:
                    predicted_block = np.ones((block_size, block_size)) * 128
            else :
                predicted_block = ref_frame[pred_y:pred_y + block_size, pred_x:pred_x + block_size]
        else:
            # Handle the case where the block is outside the boundaries
            # This might involve padding or other strategies
            predicted_block = self.handle_boundary_conditions(ref_frame, pred_y, pred_x, block_size)
        
        residual = self.generate_residual_block(current_block, predicted_block)
        
        return residual

    def inter_prediction(self, current_frame, ref_frames, block_size=None, search_range=None, nRefFrames=None, fast_me=False, mvp=(0, 0, 0)):
        start_time=time.time()
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
        input_list=[]
        if self.ParallelMode==1 or self.ParallelMode==2:
            start_time=time.time()
            for y in range(0, current_frame.shape[0], block_size):
                for x in range(0, current_frame.shape[1], block_size):
                    tuple_=(block_size, current_frame,ref_frames,x,y,search_range)
                    input_list.append(tuple_)
            ref_frames = [np.ones((self.h_pixels, self.w_pixels)) * 128]
            with Pool(8) as pool:
                results = pool.map(self.inter_prediction_parallel, input_list)
            for result in results:
                mv=result[0]
                mae=result[1]
                residual_=result[2]
                total_mae+=mae
                residual_per_block.append(residual_)
                mvs.append(mv)
            average_mae = total_mae / (len(mvs) or 1)
            print(f'time for Parallel: {self.ParallelMode} inter:{time.time()-start_time}')
            end_time=time.time()-start_time
            if self.ParallelMode == 1:
                self.inter1.append(end_time)
            else: self.inter2.append(end_time)
            return mvs, average_mae, residual_per_block
        


            

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
                                
                                if self.FMEEnable:
                                    mv, mae = self.fast_motion_estimation(current_block_vbs, ref_frames, x_vbs*2, y_vbs*2, sub_block_size, mvp, nRefFrames)
                                    residual = self.calculate_inter_frame_residual(x_vbs*2, y_vbs*2, mv, current_block_vbs, ref_frames, sub_block_size)
                                else :
                                    mv, mae = self.fast_motion_estimation(current_block_vbs, ref_frames, x_vbs, y_vbs, sub_block_size, mvp, nRefFrames)
                                    residual = self.calculate_inter_frame_residual(x_vbs, y_vbs, mv, current_block_vbs, ref_frames, sub_block_size)
                            else: 
                                
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
                        residual = self.calculate_inter_frame_residual(x, y, mv, current_block, ref_frames, block_size )
                
                if self.VBSEnable and x != 0 and y != 0:
                    RD_cost_vbs = self.calculate_RD_cost(1, 1, vbs_mae, vbs_residuals, block_size, sub_block_size, self.lam)
                    RD_cost_bs  = self.calculate_RD_cost(1, 0, mae, residual, block_size, sub_block_size, self.lam)

                    if RD_cost_bs < RD_cost_vbs:
                        mvs.append(tuple((0, mv)))
                        residual_per_block.append(tuple((0, residual)))
                    else:
                        mvs.append(tuple((1, vbs_mvs)))
                        residual_per_block.append(tuple((1, vbs_residuals)))

                    mae = vbs_mae
                else:
                    mvs.append(tuple((0, mv)))
                    residual_per_block.append(tuple((0, residual)))
                
                total_mae += mae
                mvp = mv
                
        average_mae = total_mae / (len(mvs) or 1)
        self.inter0.append(time.time()-start_time)
        return mvs, average_mae, residual_per_block

    def inter_prediction_parallel(self, x):
        block_size,current_frame,ref_frames,x,y,search_range=x
        nRefFrames=1

        mvs = []
        residual_per_block = []
        total_mae = 0.0
        
        sub_block_size = self.block_size//2
        
        #PARALLEL        
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

                    if self.fast_me:
                        
                        if self.FMEEnable:
                            mv, mae = self.fast_motion_estimation(current_block_vbs, ref_frames, x_vbs*2, y_vbs*2, sub_block_size, mvp, nRefFrames)
                            residual = self.calculate_inter_frame_residual(x_vbs*2, y_vbs*2, mv, current_block_vbs, ref_frames, sub_block_size)
                        else :
                            mv, mae = self.fast_motion_estimation(current_block_vbs, ref_frames, x_vbs, y_vbs, sub_block_size, mvp, nRefFrames)
                            residual = self.calculate_inter_frame_residual(x_vbs, y_vbs, mv, current_block_vbs, ref_frames, sub_block_size)
                    else: 
                        
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

        if self.fast_me and self.ParallelMode!=1:
            mvp=(0,0,0)
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
                residual = self.calculate_inter_frame_residual(x, y, mv, current_block, ref_frames, block_size )
        
        if self.VBSEnable and x != 0 and y != 0:
            RD_cost_vbs = self.calculate_RD_cost(1, 1, vbs_mae, vbs_residuals, block_size, sub_block_size, self.lam)
            RD_cost_bs  = self.calculate_RD_cost(1, 0, mae, residual, block_size, sub_block_size, self.lam)

            if RD_cost_bs < RD_cost_vbs:
                mvs.append(tuple((0, mv)))
                residual_per_block.append(tuple((0, residual)))
            else:
                mvs.append(tuple((1, vbs_mvs)))
                residual_per_block.append(tuple((1, vbs_residuals)))

            mae = vbs_mae
        else:
            mvs.append(tuple((0, mv)))
            residual_per_block.append(tuple((0, residual)))
                
        
        return mvs[-1], mae, residual_per_block[-1]


    def find_best_match(self, current_block, ref_frames, x, y, block_size=None, search_range=None):
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
                    # Check reference block within the current reference frame boundaries
                    if 0 <= x+dx < ref_frame.shape[1] - block_size and 0 <= y+dy < ref_frame.shape[0] - block_size:
                        
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

        return best_mv, best_mae
        
    def fast_motion_estimation(self, current_block, ref_frames, x, y, block_size, mvp, nRefFrames):
        best_mae = float('inf')
        best_mv = mvp
        best_ref_idx = 0
        
        for ref_idx, ref_frame in enumerate(ref_frames[:nRefFrames]):
            # Check all positions around the MVP
            for dx in range(mvp[0] - 1, mvp[0] + 2):
                for dy in range(mvp[1] - 1, mvp[1] + 2):
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
        transformed_block = np.round(transformed_block).astype(int)
        return transformed_block

    # Quantize the transform coefficients (TC) using the Q matrix.
    def quantize_TC(self, TC, Q):
        QTC = np.round(TC / Q).astype(int)
        return QTC

    # Round off the residual error to the nearest 2s power
    def approximate_residual_block(self, arr, n):
        # Handle negative values
        is_negative = arr < 0
        arr_abs = np.abs(arr)

        # Calculate the nearest power of 2 for positive, non-zero values
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
    def reconstruct_frame(self, mvs, ref_frames, approximated_residual_blocks, Qp_per_row, block_size):
        reconstructed_frame = np.zeros_like(ref_frames[0]).astype(np.uint8)
        if self.FMEEnable:
            ref_frames_fme=self.frac_me_reference_frame(ref_frames, block_size)
        
        for idx, mv in enumerate(mvs):
            
            if self.RCFlag != None and self.RCFlag > 0:
                if idx%self.num_blocks_per_row == 0:
                    self.set_Qp(Qp_per_row[int(idx//self.num_blocks_per_row)])

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

                    else: predicted_block = ref_frame[pred_y:pred_y + block_size, pred_x:pred_x + block_size]
                else:
                    # Handle the case where the block is outside the boundaries
                    # This might involve padding or other strategies
                    predicted_block = self.handle_boundary_conditions(ref_frame, pred_y, pred_x, block_size)
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
                        if self.FMEEnable:
                            if 0 <= pred_x+block_size < ref_frame.shape[1] - block_size and 0 <= pred_y+block_size < ref_frame.shape[0] - block_size:
                                predicted_block = ref_frame[pred_y:pred_y + block_size:2, pred_x:pred_x + block_size:2]
                            else:
                                predicted_block = np.ones((block_size//2, block_size//2)) * 128
                        else: predicted_block = ref_frame[pred_y:pred_y + block_size//2, pred_x:pred_x + block_size//2]
                    else:
                        # Handle the case where the block is outside the boundaries
                        # This might involve padding or other strategies
                        predicted_block = self.handle_boundary_conditions(ref_frame, pred_y, pred_x, block_size//2)
                    
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
        return Q

    # Edit Qp
    def set_Qp (self, Qp):
        # Edit Qp and the Corresponding Q matrix
        self.Qp = Qp
        self.Q = self.generate_Q_matrix(self.block_size, Qp)
        
        # Edit Qpm1 and the Corresponding Qm1 matrix
        if Qp > 0:
            self.Qpm1 = self.Qp-1
            self.Qm1  = self.generate_Q_matrix(self.sub_block_size, self.Qpm1)
        else:
            self.Qpm1 = self.Qp
            self.Qm1  = self.generate_Q_matrix(self.sub_block_size, self.Qpm1)

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
            else: # inter frame
                bit_rate = 8 * 2 # In inter frame we send 2 integers (8 bits) values as motion vector
            bit_rate = bit_rate + 8 * len(entropy_encoded_residual)
        else: # Calculate RD cost for the sub blocks
            if frame_type == 0: # Intra frame
                bit_rate = 8 * 4 # We will have to send 4 short ints since there are 4 sub blocks
            else: # inter frame
                bit_rate = 8 * 4 * 2 # We will have to send 4 pairs of short ints since there are 4 sub blocks
            
            for i in range(0, len(residuals)):
                entropy_encoded_residual = self.entropy_encoder_block(self.quantize_TC(self.apply_2d_dct(residuals[i]), self.Qm1), sub_block_size)
                bit_rate = bit_rate + 8 * len(entropy_encoded_residual)
        
        return lam * bit_rate + mae 
    
    def intra_prediction_parallel(self,x):
        current_frame, mode, block_size, search_range,y = x
        mvs=[]
        residual_per_block=[]
        total_mae=0.0
        ref_frame = np.ones((288, 352)) * 128
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
                        vbs_mae = vbs_mae + mae
                
                vbs_mae = vbs_mae/len(vbs_mvs)

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
                    mvs.append(tuple((0, mv)))
                    residual_per_block.append(tuple((0, residual)))
                else:
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
            total_mae += mae
        
        return mvs,total_mae,residual_per_block

                

    def intra_prediction(self, current_frame, mode=0, block_size=None, search_range=None):
        start_time=time.time()
        end_time=0
        if block_size == None: block_size = self.block_size
        if search_range == None: search_range = self.search_range

        mvs = []
        residual_per_block = []
        total_mae = 0.0

        ref_frame = np.ones((288, 352)) * 128
        if self.ParallelMode == 2:
            
            input=[]
            for y in range(0, current_frame.shape[0], block_size):
                x=(current_frame, mode, block_size, search_range,y)
                input.append(x)
            with Pool(8) as pool:
                results = pool.map(self.intra_prediction_parallel,input)
            for result in results:
                mv_row=result[0]
                mae_row=result[1]
                residual_row=result[2]
                total_mae+=mae_row
                residual_per_block=residual_per_block+residual_row
                mvs=mvs+mv_row
            average_mae = total_mae / (len(mvs) or 1)
            print('exiting intra for Parallel: ',self.ParallelMode,'with time: ',time.time()-start_time )
            end_time=time.time()-start_time
            self.intra2.append(end_time)

            return mvs, average_mae, residual_per_block, ref_frame


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
                            vbs_mae = vbs_mae + mae
                    
                    vbs_mae = vbs_mae/len(vbs_mvs)

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
                        mvs.append(tuple((0, mv)))
                        residual_per_block.append(tuple((0, residual)))
                    else:
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

                total_mae += mae

        average_mae = total_mae / (len(mvs) or 1)
        end_time=time.time()-start_time
        self.intra0.append(end_time)
        

        return mvs, average_mae, residual_per_block, ref_frame
    
    # Reconstruct Frame for intra prediction
    def reconstruct_frame_intra(self, mode, mvs, approximated_residual_blocks_per_frame, Qp_per_row, block_size):
        reconstructed_frame      = np.ones((self.h_pixels, self.w_pixels)) * 128
        residual_frame           = np.ones((self.h_pixels, self.w_pixels)) * 0

        index = 0

        rescale_inv_trans_residuals = []

        for num_block, block in enumerate(approximated_residual_blocks_per_frame):
            
            if self.RCFlag != None and self.RCFlag > 0:
                if num_block%self.num_blocks_per_row == 0:
                    self.set_Qp(Qp_per_row[int(num_block//self.num_blocks_per_row)])
            
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

                reconstructed_frame[y:y+block_size, x:x+block_size] = reconstructed_block
                index = index + 1

        reconstructed_frame = reconstructed_frame.astype(np.uint8)
        residual_frame      = residual_frame

        return reconstructed_frame, residual_frame

    def differential_encoder_frame(self, frame_type, mv_for_frame, Qp_for_frame):
        
        frame_mvs    = ""
        ref_intra_mv = 0
        ref_inter_mv = (0, 0, 0)  # Including reference frame index
        ref_Qp       = 0

        if frame_type == 0: # Intra frame
            for j, mv in enumerate(mv_for_frame):

                if mv[0] == 0: # No split
                    diff_mv = mv[1] - ref_intra_mv
                    
                    if self.RCFlag != None and self.RCFlag > 0 and j%self.num_blocks_per_row == 0:
                        diff_qp = Qp_for_frame[int(j//self.num_blocks_per_row)] - ref_Qp

                    if j == 0:
                        if self.RCFlag != None and self.RCFlag > 0 and j%self.num_blocks_per_row == 0:
                            frame_mvs = str(diff_qp)+"@0\'(" + str(diff_mv) + ")"
                        else:
                            frame_mvs = "0\'(" + str(diff_mv) + ")"
                    else:
                        if self.RCFlag != None and self.RCFlag > 0 and j%self.num_blocks_per_row == 0:
                            frame_mvs += ";" + str(diff_qp) + "@0\'(" + str(diff_mv) + ")"
                        else:
                            frame_mvs += ";0\'(" + str(diff_mv) + ")"
                    
                    ref_intra_mv = mv[1]
                    
                    if self.RCFlag != None and self.RCFlag > 0 and j%self.num_blocks_per_row == 0:
                        ref_Qp = Qp_for_frame[int(j//self.num_blocks_per_row)]

                elif mv[0] == 1: #Split
                    
                    if self.RCFlag != None and self.RCFlag > 0 and j%self.num_blocks_per_row == 0:
                        diff_qp = Qp_for_frame[int(j//self.num_blocks_per_row)] - ref_Qp
                    
                    for k, sb_mv in enumerate(mv[1]): # loop through the subblock MV
                        diff_mv = sb_mv - ref_intra_mv
                        if k == 0:
                            if self.RCFlag != None and self.RCFlag > 0 and j%self.num_blocks_per_row == 0:
                                frame_mvs += ";" + str(diff_mv) + "@1\'(" + str(diff_mv) + ","
                            else:
                                frame_mvs += ";1\'(" + str(diff_mv) + ","
                        elif k == 3:
                            frame_mvs += str(diff_mv) + ")"
                        else:
                            frame_mvs +=  str(diff_mv) + ","
                        
                        ref_intra_mv = sb_mv
                    
                    if self.RCFlag != None and self.RCFlag > 0 and j%self.num_blocks_per_row == 0:
                        ref_Qp = Qp_for_frame[int(j//self.num_blocks_per_row)]

        else: # Inter frame
            for j, mv in enumerate(mv_for_frame):
                # Calculate differential for x, y and reference frame index
                if mv[0] == 0: # No split
                    diff_mv = (mv[1][0] - ref_inter_mv[0], mv[1][1] - ref_inter_mv[1], mv[1][2] - ref_inter_mv[2])
                    
                    if self.RCFlag != None and self.RCFlag > 0 and j%self.num_blocks_per_row == 0:
                        diff_qp = Qp_for_frame[int(j//self.num_blocks_per_row)] - ref_Qp
                    
                    if j == 0:
                        if self.RCFlag != None and self.RCFlag > 0 and j%self.num_blocks_per_row == 0:
                            frame_mvs = str(diff_qp) + "@0\'" + str(diff_mv)
                        else:
                            frame_mvs = "0\'" + str(diff_mv)
                    else:
                        
                        if self.RCFlag != None and self.RCFlag > 0 and j%self.num_blocks_per_row == 0:
                            frame_mvs += ";" + str(diff_qp) + "@0\'" + str(diff_mv)
                        else:
                            frame_mvs += ";0\'" + str(diff_mv)
                    ref_inter_mv = mv[1]
                    
                    if self.RCFlag != None and self.RCFlag > 0 and j%self.num_blocks_per_row == 0:
                        ref_Qp = Qp_for_frame[int(j//self.num_blocks_per_row)]
                
                elif mv[0] == 1: # Split
                    
                    if self.RCFlag != None and self.RCFlag > 0 and j%self.num_blocks_per_row == 0:
                        diff_qp = Qp_for_frame[int(j//self.num_blocks_per_row)] - ref_Qp
                    
                    for k, sb_mv in enumerate(mv[1]):
                        diff_mv = (sb_mv[0] - ref_inter_mv[0], sb_mv[1] - ref_inter_mv[1], sb_mv[2] - ref_inter_mv[2])
                        if k == 0:
                            if self.RCFlag != None and self.RCFlag > 0 and j%self.num_blocks_per_row == 0:
                                frame_mvs += ";" + str(diff_qp) +"@1\'(" + str(diff_mv) + ","
                            else:
                                frame_mvs += ";1\'(" + str(diff_mv) + ","
                        elif k == 3:
                            frame_mvs += str(diff_mv) + ")"
                        else:
                            frame_mvs += str(diff_mv) + ","
                        
                        ref_inter_mv = sb_mv
                    
                    if self.RCFlag != None and self.RCFlag > 0 and j%self.num_blocks_per_row == 0:
                        ref_Qp = Qp_for_frame[int(j//self.num_blocks_per_row)]
            
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

        mvs_per_frame        = self.encoded_package["MVS per Frame"]
        residual_per_frame   = self.encoded_package["approx residual"]
        Qp_per_row_per_frame = self.encoded_package["Qp_per_row_per_frame"]
        frame_type_seq       = self.encoded_package["frame_type_seq"]

        f_trns_mvs_per_frame     = open(mv_file, "w")
        f_trns_mvs_per_frame_raw = open(f"files/mvs_per_frame_raw.txt", "w")

        f_trns_res_per_frame     = open(residual_file, "w")
        for i in range(0, len(mvs_per_frame)):
            mvs = mvs_per_frame[i]
            Qp_per_row = Qp_per_row_per_frame[i]
            residuals = residual_per_frame[i]
            frame_type = frame_type_seq[i]
            f_trns_mvs_per_frame.write(str(frame_type) + "|" + self.differential_encoder_frame(frame_type, mvs, Qp_per_row) + "\n")
            f_trns_mvs_per_frame_raw.write(str(frame_type) + "|" + str(mvs) + "\n")
            f_trns_res_per_frame.write(str(residuals) + "\n")
        
        f_trns_mvs_per_frame.close()
        f_trns_mvs_per_frame_raw.close()
        f_trns_res_per_frame.close()

    # Find appropriate Qp values from the Qp tables
    def get_appropriate_Qp_value(self, frame_type, row_bit_budget): # Frame type: 0 == Intra, 1 == Inter
        num_qps = len(self.qr_rate_tables[frame_type])
        for Qp, bitrate in enumerate(self.qr_rate_tables[frame_type]):
            if bitrate < row_bit_budget:
                return Qp, bitrate

    def complete_intra_flow(self, current_padded_frame, intra_mode, block_size, search_range, generate_row_wise_stats=True):
        Qp_per_row       = []
        quantized_blocks = []

        mvs, average_mae, intra_residual, _ = self.intra_prediction(current_padded_frame, intra_mode, block_size, search_range)
        
        # Number of bits we can spend on a row
        row_bit_budget = self.bitrate_per_row
        bits_spent = 0

        quantized_sized = 0

        bits_spent_per_row = [0]
        bits_spent_per_row_percentage = []

        for num_block, block in enumerate(intra_residual): # block is a tuple : (split, residual/s)
            # Adjust the row bit budget for current row based on bit spent on the previous row
            if self.RCFlag != None and self.RCFlag > 0:
                if num_block == 0:
                    row_bit_budget = self.bitrate_per_row
                    Qp_used, bits_spent = self.get_appropriate_Qp_value(0, row_bit_budget)
                    self.set_Qp(Qp_used)
                    Qp_per_row.append(Qp_used)
                elif num_block%self.num_blocks_per_row == 0:
                    row_bit_budget = self.bitrate_per_row + (row_bit_budget - bits_spent)
                    Qp_used, bits_spent = self.get_appropriate_Qp_value(0, row_bit_budget)
                    self.set_Qp(Qp_used)
                    Qp_per_row.append(Qp_used)

            if block[0] == 0: # No split
                transformed_block = self.apply_2d_dct(block[1])
                quantized_block   = self.quantize_TC(transformed_block, self.Q)
                #print("Entropy: ", self.entropy_encoder_block(quantized_block, block_size))
                quantized_sized   += len(self.entropy_encoder_block(quantized_block, block_size))
                quantized_blocks.append(tuple((0, quantized_block)))
            else: # Split
                vbs_blocks = []
                
                for sub_block in block[1]:
                    transformed_block = self.apply_2d_dct(sub_block)
                    quantized_block   = self.quantize_TC(transformed_block, self.Qm1)
                    quantized_sized   += len(self.entropy_encoder_block(quantized_block, block_size//2))
                    vbs_blocks.append(quantized_block)
                
                quantized_blocks.append(tuple((1, vbs_blocks)))
            
            if generate_row_wise_stats and (num_block+1)%self.num_blocks_per_row == 0:
                bits_spent_per_row.append(quantized_sized)
        
        reconstructed_frame, residual_frame = self.reconstruct_frame_intra(intra_mode, mvs, quantized_blocks, Qp_per_row, block_size)

        if generate_row_wise_stats:
            temp_list = []
            
            for i in range(1, len(bits_spent_per_row)):
                temp_list.append(bits_spent_per_row[i] - bits_spent_per_row[i-1])

            for row in temp_list:
                bits_spent_per_row_percentage.append((row/quantized_sized) * 100)

        return mvs, average_mae, quantized_blocks, Qp_per_row, reconstructed_frame, residual_frame, quantized_sized, bits_spent_per_row_percentage

    def complete_inter_flow(self, current_padded_frame, ref_frames, block_size, search_range, generate_row_wise_stats=True):
        quantized_blocks                        = []
        Qp_per_row                              = []

        if self.FMEEnable:
            mvs, average_mae, inter_residual    = self.inter_prediction(current_padded_frame, self.frac_me_reference_frame(ref_frames, block_size), block_size, search_range*2, fast_me=self.fast_me)      # Inter prediction
        else :
            mvs, average_mae, inter_residual    = self.inter_prediction(current_padded_frame, ref_frames, block_size, search_range, fast_me=self.fast_me)      # Inter prediction
        
        if self.FMEEnable:
            ref_frames_fme=self.frac_me_reference_frame(ref_frames, block_size)
        
        # Number of bits we can spend on a row
        row_bit_budget = self.bitrate_per_row
        bits_spent = 0
        
        quantized_sized = 0
        
        bits_spent_per_row = [0]
        bits_spent_per_row_percentage = []
        
        for num_block, block in enumerate(inter_residual): # block is a tuple : (split, residual/s)
            
            # Adjust the row bit budget for current row based on bit spent on the previous row
            if self.RCFlag != None and self.RCFlag > 0:
                if num_block == 0:
                    row_bit_budget = self.bitrate_per_row
                    Qp_used, bits_spent = self.get_appropriate_Qp_value(0, row_bit_budget)
                    self.set_Qp(Qp_used)
                    Qp_per_row.append(Qp_used)
                elif num_block%self.num_blocks_per_row == 0:
                    row_bit_budget = self.bitrate_per_row + (row_bit_budget - bits_spent)
                    Qp_used, bits_spent = self.get_appropriate_Qp_value(0, row_bit_budget)
                    self.set_Qp(Qp_used)
                    Qp_per_row.append(Qp_used)
            
            if block[0] == 0:
                transformed_block               = self.apply_2d_dct(block[1])
                quantized_transform             = self.quantize_TC(transformed_block, self.Q)
                quantized_sized                 += len(self.entropy_encoder_block(quantized_transform, block_size))
                quantized_blocks.append(tuple((0,quantized_transform)))
            else:
                vbs_blocks = [] 

                for sub_block in block[1]:
                    transformed_block = self.apply_2d_dct(sub_block)
                    quantized_transform = self.quantize_TC(transformed_block, self.Qm1)
                    quantized_sized   += len(self.entropy_encoder_block(quantized_transform, block_size//2))
                    vbs_blocks.append(quantized_transform)

                quantized_blocks.append(tuple((1, vbs_blocks)))
            
            if generate_row_wise_stats and (num_block+1)%self.num_blocks_per_row == 0:
                bits_spent_per_row.append(quantized_sized)
        reconstructed_frame = self.reconstruct_frame(mvs, ref_frames, quantized_blocks, Qp_per_row, block_size)
        
        if generate_row_wise_stats:
            temp_list = []
            
            for i in range(1, len(bits_spent_per_row)):
                temp_list.append(bits_spent_per_row[i] - bits_spent_per_row[i-1])

            for row in temp_list:
                bits_spent_per_row_percentage.append((row/quantized_sized) * 100)

        return mvs, average_mae, quantized_blocks, Qp_per_row, reconstructed_frame, quantized_sized, bits_spent_per_row_percentage


    def encode_frames_parallel(self, data):
        start_time=time.time()
        i=data[1]
        q=data[0]
        
        ref_frames = q.get()
        
        while True:
            
            if i!=len(ref_frames)-1:
                q.put(ref_frames)
            else: break
                    
        new_ref_frame = ref_frames
        frame_type_seq = []
        mae_per_frame  = []
        mvs_per_frame  = []
        approximated_residual_blocks_per_frame = []
        Qp_per_row_per_frame = []
        reconstructed_frames = []
        frame_no       = []
        psnr_per_frame = []
        ssim_per_frame = []

        search_range = self.search_range
        block_size   = self.block_size
        intra_dur    = self.intra_dur
        intra_mode   = self.intra_mode

        frame_no.append(i)
        nref_frames          = []
        mvs_frame            = []
        current_padded_frame = self.pad_hw(self.y_only_f_arr[i], block_size, 128)
        quantized_blocks     = []
        Qp_per_row           = []
        mvs=[]

        
        if i%intra_dur == 0 and self.ParallelMode !=1 :   # Intra
            self.set_Qp(self.const_init_Qp)
            mvs, average_mae, quantized_blocks, Qp_per_row, reconstructed_frame, _ , residual_size, row_wise_stats = self.complete_intra_flow(current_padded_frame, intra_mode, block_size, search_range)
            frame_type_seq.append(0)
        else:                  # Inter
            self.set_Qp(self.const_init_Qp)
            mvs, average_mae, quantized_blocks, Qp_per_row, reconstructed_frame, residual_size, row_wise_stats = self.complete_inter_flow(current_padded_frame, ref_frames, block_size, search_range)
            frame_type_seq.append(1)

            if self.RCFlag != None and self.RCFlag > 1:
                if residual_size > self.intra_thresh:
                    mvs, average_mae, quantized_blocks, Qp_per_row, reconstructed_frame, _ , residual_size, row_wise_stats = self.complete_intra_flow(current_padded_frame, intra_mode, block_size, search_range)
                    frame_type_seq.pop()
                    frame_type_seq.append(0)
            
        
        mvs_per_frame.append(mvs)
        mae_per_frame.append(average_mae)
        reconstructed_frames.append(reconstructed_frame)
        approximated_residual_blocks_per_frame.append(quantized_blocks)
        Qp_per_row_per_frame.append(Qp_per_row)
        avg_psrn, avg_ssim = self.calculate_metrics(self.y_only_f_arr[i], reconstructed_frame)
        psnr_per_frame.append(avg_psrn)
        ssim_per_frame.append(avg_ssim)
        new_ref_frame.append(reconstructed_frame)
        q.put(new_ref_frame)
        if  i%intra_dur == 0 :
                print('Intra: ',time.time()-start_time)
                self.intra3.append(time.time()-start_time)
        else: 
            print ('Inter:', time.time()-start_time)
            self.inter3.append(time.time()-start_time)

        return frame_type_seq[-1], quantized_blocks, Qp_per_row_per_frame[-1],mvs_per_frame[-1],reconstructed_frame


    def encode(self, intra_mode=None, intra_dur=None, search_range=None, block_size=None, save_enc_pkg=True):
        # global global_ref_buffer
        encoded_package = {}
        frame_type_seq       = []
        mae_per_frame  = []
        mvs_per_frame  = []
        approximated_residual_blocks_per_frame = []
        Qp_per_row_per_frame = []
        ref_frames = [np.ones((self.h_pixels, self.w_pixels)) * 128]  # Start with one reference frame filled with 128
        reconstructed_frames = []
        frame_no       = []
        psnr_per_frame = []
        ssim_per_frame = []

        if search_range == None: search_range = self.search_range
        if block_size   == None: block_size   = self.block_size
        if intra_dur    == None: intra_dur    = self.intra_dur
        if intra_mode   == None: intra_mode   = self.intra_mode



        if self.ParallelMode == 3:
            m = mp.Manager()
            queue = m.Queue()
            print("queue:", queue)
            queue.put([np.ones((self.h_pixels, self.w_pixels)) * 128])
            with Pool(8) as pool:
                results = pool.map(self.encode_frames_parallel, [(queue, arg) for arg in range(self.frames)])
            for result in results:
                frame_type_seq_, quantized_blocks, Qp_per_row, mvs,reconstructed_frame = result
                Qp_per_row_per_frame.append(Qp_per_row)
                frame_type_seq.append(frame_type_seq_)
                reconstructed_frames.append(reconstructed_frame)
                approximated_residual_blocks_per_frame.append(quantized_blocks)
                mvs_per_frame.append(mvs)
            
            reconstructed_frames=np.array(reconstructed_frames)
            
        else:
            for i in range(self.frames):
                frame_no.append(i)
                nref_frames          = []
                mvs_frame            = []
                current_padded_frame = self.pad_hw(self.y_only_f_arr[i], block_size, 128)
                quantized_blocks     = []
                Qp_per_row           = []
                
                if i%intra_dur == 0 and self.ParallelMode !=1 :     # Intra
                    self.set_Qp(self.const_init_Qp)
                    mvs, average_mae, quantized_blocks, Qp_per_row, reconstructed_frame, _ , residual_size, row_wise_stats = self.complete_intra_flow(current_padded_frame, intra_mode, block_size, search_range)
                    frame_type_seq.append(0)
                else:                                               # Inter
                    self.set_Qp(self.const_init_Qp)
                    if self.ParallelMode == 1 or self.ParallelMode == 2:
                        if self.ParallelMode == 1: ref_frames = [np.ones((self.h_pixels, self.w_pixels)) * 128]
                    mvs, average_mae, quantized_blocks, Qp_per_row, reconstructed_frame, residual_size, row_wise_stats = self.complete_inter_flow(current_padded_frame, ref_frames, block_size, search_range)
                    frame_type_seq.append(1)

                    if self.RCFlag != None and self.RCFlag > 1:
                        if residual_size > self.intra_thresh:
                            mvs, average_mae, quantized_blocks, Qp_per_row, reconstructed_frame, _ , residual_size, row_wise_stats = self.complete_intra_flow(current_padded_frame, intra_mode, block_size, search_range)
                            frame_type_seq.pop()
                            frame_type_seq.append(0)
                
                mvs_per_frame.append(mvs)
                mae_per_frame.append(average_mae)
                reconstructed_frames.append(reconstructed_frame)
                approximated_residual_blocks_per_frame.append(quantized_blocks)
                Qp_per_row_per_frame.append(Qp_per_row)

                if i < self.frames - 1:  # Set the current reconstructed frame as the reference for the next one
                    if len(ref_frames) >= self.nRefFrames:
                        ref_frames.pop(0)  # Remove the oldest reference frame if we've reached the limit
                    ref_frames.append(reconstructed_frame)

                avg_psrn, avg_ssim = self.calculate_metrics(self.y_only_f_arr[i], reconstructed_frame)
                psnr_per_frame.append(avg_psrn)
                ssim_per_frame.append(avg_ssim)
       
        decoded_frames = self.decoder.decode(frame_type_seq, approximated_residual_blocks_per_frame, Qp_per_row_per_frame, mvs_per_frame, intra_mode, intra_dur, block_size, self.frames, self.w_pixels, self.h_pixels)

        encoded_package["block size"]           = block_size
        encoded_package["num frames"]           = self.frames
        encoded_package["height in pixels"]     = self.h_pixels
        encoded_package["width in pixels"]      = self.w_pixels
        encoded_package["search range"]         = search_range
        encoded_package["PSNR per frame"]       = psnr_per_frame
        encoded_package["SSIM per frame"]       = ssim_per_frame
        encoded_package["MAE per Frame"]        = mae_per_frame
        encoded_package["MVS per Frame"]        = mvs_per_frame
        encoded_package["approx residual"]      = approximated_residual_blocks_per_frame
        encoded_package["Qp_per_row_per_frame"] = Qp_per_row_per_frame
        encoded_package["frame_type_seq"]       = frame_type_seq
        self.encoded_package_f                  = True

        if save_enc_pkg == True:
            self.encoded_package = encoded_package

        self.save_y_only(f"yuv/y_only_reconstructed.yuv", reconstructed_frames)

        # Comparing the decoded frames with the reconstructed frames
       
        print(f'0: Intra= {self.intra0}\n0: Inter= {self.inter0}\n1: Intra=  {self.intra1}\n1: Inter= {self.inter1}\n2: Intra= {self.intra2}\n2: Inter= {self.inter2}\n3: Intra=  {self.intra3}\n3: Inter= {self.inter3}')
        return psnr_per_frame