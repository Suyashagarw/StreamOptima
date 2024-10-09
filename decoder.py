import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

class decoder:

    def __init__(self, intra_mode, intra_dur, block_size, frames, height, width, Qp, nRefFrames, FMEEnable, lam, VBSEnable, VBSoverlay=None,RCflag=False):
        self.intra_mode    = intra_mode
        self.intra_dur     = intra_dur
        self.block_size    = block_size
        self.sub_block_size= block_size//2
        self.frames        = frames
        self.h_pixels      = height
        self.w_pixels      = width
        self.ref_frame     = np.ones((height, width)) * 128 # Hypothetical reference frame where every pixel is 128 
        self.decoded_vid   = None
        self.decoded_vid_f = False
        self.Qp            = Qp
        self.Qpm1          = None
        self.Q             = self.generate_Q_matrix(block_size, Qp)
        self.Qm1           = None
        self.nRefFrames    = nRefFrames
        self.FMEEnable     = FMEEnable
        self.lam           = lam
        self.VBSEnable     = VBSEnable
        self.VBSoverlay    = VBSoverlay
        self.qp_table      = np.zeros((frames,height//block_size))
        self.RCflag        = RCflag
        # print(self.RCflag)

        if Qp > 0:
            self.Qpm1 = self.Qp-1
            self.Qm1  = self.generate_Q_matrix(self.sub_block_size, self.Qpm1)
        else:
            self.Qpm1 = self.Qp
            self.Qm1  = self.generate_Q_matrix(self.sub_block_size, self.Qpm1)
    
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
        self.Qp = Qp
        self.Q = self.generate_Q_matrix(self.block_size, Qp)
    
    # Reconstruct the block by adding the approximated residual to the predicted block."""
    def reconstruct_block(self, predicted_block, residual_block, Q):
        rescale_quant_trans = self.rescale_QTC(residual_block, Q)
        inv_transform_block = self.apply_2d_idct(rescale_quant_trans)
        return (predicted_block + inv_transform_block).astype(np.uint8)

    def construct_VBS_overlay(self, split, block):
        overlay_reconstructed_block = np.copy(block)
        overlay_reconstructed_block[0, :]  = 0
        #overlay_reconstructed_block[7,:]   = 0
        overlay_reconstructed_block[:,0]   = 0
        #overlay_reconstructed_block[:,7]   = 0
        
        if split == 1:
            overlay_reconstructed_block[self.block_size//2]    = 0
            overlay_reconstructed_block[:,self.block_size//2] = 0
        
        return overlay_reconstructed_block
    
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

    # Inter prediction: Decode a frame using the provided motion vectors and residuals.
    def decode_frame_inter(self, ref_frames, mvs, approximated_residual_blocks, block_size=None,frame_no=99):
        if block_size == None: block_size = self.block_s

        reconstructed_frame = np.zeros_like(ref_frames[0]).astype(np.uint8)
        
        if self.FMEEnable:
            ref_frames_fme=self.frac_me_reference_frame(ref_frames, block_size)
        
        if self.VBSoverlay:
            overlay_reconstructed_frame = np.zeros_like(ref_frames[0]).astype(np.uint8)
        else:
            overlay_reconstructed_frame = None

        for idx, mv in enumerate(mvs):
            if idx%(self.w_pixels/self.block_size) == 0 and self.RCflag:
                self.changed_qp(self.qp_table[frame_no,idx//(self.w_pixels//self.block_size)])
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
                    #predicted_block = ref_frame[pred_y:pred_y + block_size, pred_x:pred_x + block_size]
                else:
                    # Handle the case where the block is outside the boundaries
                    # This might involve padding or other strategies
                    predicted_block = self.handle_boundary_conditions(ref_frame, pred_y, pred_x, block_size)
                # print(block_y, mv[1])
                # predicted_block = ref_frames[mv[2]][block_y + mv[1]:block_y + mv[1] + block_size, block_x + mv[0]:block_x + mv[0] + block_size]
                reconstructed_block = self.reconstruct_block(predicted_block, approximated_residual_blocks[idx][1], self.Q)
                if self.VBSEnable:
                    overlay_reconstructed_block = self.construct_VBS_overlay(0, reconstructed_block)

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
                        ref_frame = ref_frames[sb_mv[2]]  # Reference frame corresponding to the third component of mv
                    #ref_frame = ref_frames[sb_mv[2]]  # Reference frame corresponding to the third component of mv
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

                    if self.VBSoverlay:
                        overlay_reconstructed_block = self.construct_VBS_overlay(1, reconstructed_block)

            reconstructed_frame[block_y:block_y+block_size, block_x:block_x+block_size] = reconstructed_block

            if self.VBSoverlay:
                overlay_reconstructed_frame[block_y:block_y+block_size, block_x:block_x+block_size] = overlay_reconstructed_block

        return reconstructed_frame, overlay_reconstructed_frame

    # Intra Prediction: Decode a frame using the provided motion vectors and residuals.
    def decode_frame_intra(self, mvs, approximated_residual_blocks_per_frame, mode=None, block_size=None,frame_no=99):
        if mode == None: mode = self.intra_mode
        if block_size == None: block_size = self.block_s

        reconstructed_frame      = np.ones((self.h_pixels, self.w_pixels)) * 128
        
        if self.VBSoverlay:
            overlay_reconstructed_frame = np.ones((self.h_pixels, self.w_pixels)) * 0
        else:
            overlay_reconstructed_frame = None
        
        residual_frame           = np.ones((self.h_pixels, self.w_pixels)) * 0

        index = 0
        
        rescale_inv_trans_residuals = []

        for idx,block in enumerate(approximated_residual_blocks_per_frame):
            # print(idx)
            if idx%(self.w_pixels/self.block_size) == 0 and self.RCflag:
                # print(idx)
                self.changed_qp(self.qp_table[frame_no,idx//(self.w_pixels//self.block_size)])
            


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
        # block_count=0
        for y in range(0, self.h_pixels, block_size):
            for x in range(0, self.w_pixels, block_size):
                
                # block_count+=1
                if x == 0 and mode == 0:
                    reconstructed_block = np.ones((block_size, block_size)) * 128 + rescale_inv_trans_residuals[index][1]
                    residual_frame[y:y+block_size, x:x+block_size] = rescale_inv_trans_residuals[index][1]
                    
                    if self.VBSoverlay:
                        overlay_reconstructed_block = self.construct_VBS_overlay(0, reconstructed_block)
                
                elif y == 0 and mode == 1:
                    reconstructed_block = np.ones((block_size, block_size)) * 128 + rescale_inv_trans_residuals[index][1]
                    residual_frame[y:y+block_size, x:x+block_size] = rescale_inv_trans_residuals[index][1]
                    
                    if self.VBSoverlay:
                        overlay_reconstructed_block = self.construct_VBS_overlay(0, reconstructed_block)
                
                elif mode == 0:
                    if rescale_inv_trans_residuals[index][0] == 0: # No Split
                        reconstructed_block = reconstructed_frame[y:y+block_size, (x+mvs[index][1]):(x+mvs[index][1])+block_size] + rescale_inv_trans_residuals[index][1]
                        residual_frame[y:y+block_size, x:x+block_size] = rescale_inv_trans_residuals[index][1]

                        if self.VBSoverlay:
                            overlay_reconstructed_block = self.construct_VBS_overlay(0, reconstructed_block)
                    
                    else: # Split
                        j = 0
                        reconstructed_block = np.ones((block_size, block_size))
                        for y_vbs in range(y, y+block_size, block_size//2):
                            for x_vbs in range(x, x+block_size, block_size//2):
                                reconstructed_block[y_vbs-y:y_vbs-y+(block_size//2), x_vbs-x:x_vbs-x+(block_size//2)] = reconstructed_frame[y_vbs:y_vbs+(block_size//2), (x_vbs+mvs[index][1][j]):(x_vbs+mvs[index][1][j] + (block_size//2))] + rescale_inv_trans_residuals[index][1][j]
                                residual_frame[y_vbs:y_vbs+block_size//2, x_vbs:x_vbs+block_size//2] = rescale_inv_trans_residuals[index][1][j]
                                j = j + 1
                            
                        if self.VBSoverlay:
                            overlay_reconstructed_block = self.construct_VBS_overlay(1, reconstructed_block)

                elif mode == 1:
                    if rescale_inv_trans_residuals[index][0] == 1: # No Split
                        reconstructed_block = reconstructed_frame[(y+mvs[index][1]):(y+mvs[index][1])+block_size, x:x+block_size] + rescale_inv_trans_residuals[index][1]
                        residual_frame[y:y+block_size, x:x+block_size] = rescale_inv_trans_residuals[index][1]
                        
                        if self.VBSoverlay:
                            overlay_reconstructed_block = self.construct_VBS_overlay(0, reconstructed_block)

                    else: # Split
                        j = 0
                        reconstructed_block = np.ones((block_size, block_size))
                        for y_vbs in range(y, y+block_size, block_size//2):
                            for x_vbs in range(x, x+block_size, block_size//2):
                                reconstructed_block[(y_vbs-y):(y_vbs-y)+(block_size//2), (x_vbs-x):(x_vbs-x)+(block_size//2)] = reconstructed_frame[(y_vbs+mvs[index][1][j]):(y_vbs+mvs[index][1][j])+block_size//2, x_vbs:x_vbs+block_size//2] + rescale_inv_trans_residuals[index][1][j]
                                residual_frame[y_vbs:y_vbs+block_size//2, x_vbs:x_vbs+block_size//2] = rescale_inv_trans_residuals[index][1][j]
                                j = j + 1
                        
                        if self.VBSoverlay:
                            overlay_reconstructed_block = self.construct_VBS_overlay(1, reconstructed_block)

                reconstructed_frame[y:y+block_size, x:x+block_size] = reconstructed_block
                index = index + 1
                
                if self.VBSoverlay:
                    overlay_reconstructed_frame[y:y+block_size, x:x+block_size] = overlay_reconstructed_block

        if self.VBSoverlay:
            overlay_reconstructed_frame = overlay_reconstructed_frame.astype(np.uint8)                    

        return reconstructed_frame.astype(np.uint8), residual_frame, overlay_reconstructed_frame
    
    # def handle_boundary_conditions(self, ref_frame, y, x, block_size):
    #     frame_height, frame_width = ref_frame.shape
    #     padded_block = np.zeros((block_size, block_size), dtype=ref_frame.dtype)

    #     # Calculate the overlap between the desired block and the actual frame
    #     y_start = max(y, 0)
    #     y_end = min(y + block_size, frame_height)
    #     x_start = max(x, 0)
    #     x_end = min(x + block_size, frame_width)

    #     # Copy the overlapping part from the reference frame to the padded block
    #     padded_block[y_start-y:y_end-y, x_start-x:x_end-x] = ref_frame[y_start:y_end, x_start:x_end]

    #     return padded_block
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
    
    # Main decoding function
    def decode(self, residual_file, mv_file, intra_mode=None, intra_dur=None, block_size = None, frames=None, width=None, height=None, save_decoded_frames=True):
        
        if intra_dur == None: intra_dur = self.intra_dur
        if intra_mode == None: intra_mode = self.intra_mode
        if block_size == None: block_size = self.block_s
        if frames == None: frames = self.frames
        if width  == None: width  = self.w_pixels
        if height == None: height = self.h_pixels

        residuals_per_frame = residual_file
        mv_per_frame = mv_file

        ref_frames = [np.ones((height, width)) * 128]  # Hypothetical reference frame of all-128-values for the first frame
        decoded_frames = []
        overlay_decoded_frames = []
        
        for i in range(frames):
            if i%intra_dur == 0:
               decoded_frame, _ , overlay_decoded_frame = self.decode_frame_intra(mv_per_frame[i], residuals_per_frame[i], intra_mode, block_size,i)
               #plt.imshow(decoded_frame)
               #plt.show()
               #exit()
               ref_frames = []
            else: 
                #if self.FMEEnable:
                #   ref_frames= self.frac_me_reference_frame(ref_frames, block_size)

               decoded_frame, overlay_decoded_frame  = self.decode_frame_inter(ref_frames, mv_per_frame[i], residuals_per_frame[i], block_size,i)
            
            decoded_frames.append(decoded_frame)
            
            if self.VBSoverlay:
                overlay_decoded_frames.append(overlay_decoded_frame)

            if i < frames - 1:  # Set the current reconstructed frame as the reference for the next one
                if len(ref_frames) >= self.nRefFrames:
                    ref_frames.pop(0)  # Remove the oldest reference frame if we've reached the limit
                ref_frames.append(decoded_frame)

        if save_decoded_frames:
            self.decoded_vid_f = True
            self.decoded_vid   = decoded_frames
            
            if self.VBSoverlay:
                self.overlay_decoded_vid = overlay_decoded_frames

        return decoded_frames
    
    # Entropy decoder for Residual blocks
    def entropy_decoder_block(self, encoded_block, block_size):
        arr=[]
        i=0
        n=block_size
        while i < (len(encoded_block)):
            if encoded_block[i] < 0:
                count=-encoded_block[i]
                for pos in range(1,count+1):
                    arr.append(encoded_block[i+pos])
                i=i+pos
                # print('i:{}'.format(i))
            else:
                # print('asdfa:',i)
                if encoded_block[i]==0:
                    break
                count=encoded_block[i]
                for pos in range(count):
                    arr.append(0)
            i+=1

        x=np.zeros((block_size,block_size), dtype=int)
        result=x.tolist()
        pos=0
        for k in range(2 * n - 1):
            if k < n:
                i, j = 0, k
            else:
                i, j = k - n + 1, n - 1
            while i < n and j >= 0:
                if(pos==len(arr)):
                    break
                # print(i,j)
                result[i][j]=arr[pos]
                pos+=1
                i += 1
                j -= 1
            if pos==len(arr):
                break
        return result
    

    # Differntial decoder for MV
    def differential_decoder_frame(self, mv_for_frame):
        raw = mv_for_frame.split("|")
        frame_type = int(raw[0])
        mvs = raw[1].split(";")

        frame_mvs = []
        
        if frame_type == 0: # Intra 
            ref_mv = 0 
            for j, mv_split in enumerate(mvs):
                split, mv_b = mv_split.split("\'")
                if split == "0": # No Split
                    mv = ref_mv + int(eval(mv_b))
                    frame_mvs.append(tuple((0,mv)))
                    ref_mv = mv
                elif split == "1": #Split
                    mv_b = eval(mv_b)
                    sb_mv_list = []
                    for sb_mv in mv_b:
                        mv = ref_mv + sb_mv
                        sb_mv_list.append(mv)
                        ref_mv = mv
                    frame_mvs.append(tuple((1, sb_mv_list)))
        else:               # Inter
            ref_mv = (0,0,0)
            for j, mv_split in enumerate(mvs):
                split, mv_b = mv_split.split("\'")
                if split == "0": # No split
                    mv_i = eval(mv_b)
                    mv = (ref_mv[0] + mv_i[0], ref_mv[1] + mv_i[1], ref_mv[2] + mv_i[2])
                    frame_mvs.append(tuple((0,mv)))
                    ref_mv = mv
                elif split == "1": # Split
                    mv_b = eval(mv_b)
                    sb_mv_list = []
                    for sb_mv in mv_b:
                        mv = (ref_mv[0] + sb_mv[0], ref_mv[1] + sb_mv[1], ref_mv[2] + sb_mv[2])
                        sb_mv_list.append(mv)
                        ref_mv = mv
                    
                    frame_mvs.append(tuple((1, sb_mv_list)))
        
        return frame_mvs

    def entropy_decoder_frame(self, residual_for_frame, block_size):
        residuals_per_block = residual_for_frame.split(";")
        
        decoded_residuals = []

        for i, split_residual in enumerate(residuals_per_block):
            split, residual = split_residual.split("\'")
            if split == "0": # No split
                entropy_res = eval(residual)
                decoded_residuals.append(tuple((0, np.array(self.entropy_decoder_block(entropy_res, block_size)))))
            elif split == "1": # Split
                entropy_res = eval(residual)
                sb_entropy_res_list = []

                for j, sb_entropy_res in enumerate(entropy_res):
                    sb_entropy_res_list.append(np.array(self.entropy_decoder_block(sb_entropy_res, block_size//2)))
                
                decoded_residuals.append(tuple((1, sb_entropy_res_list)))

        return decoded_residuals


    def decode_differential_entropy(self, all_mv_f, all_residual_f, block_size):
        mv_for_vid  = []
        res_for_vid = []
        
        with open(all_mv_f) as all_mv:
            for line in all_mv:
                mv_for_vid.append(self.differential_decoder_frame(line))
        # print(mv_for_vid)
        #f_decode_trns_mvs_per_frame     = open("decode_trns_mvs_per_frame.txt", "a")

        #for i in range(0, len(mv_for_vid)):
        #    frame_type = mv_for_vid[i][0]
        #    mvs        = mv_for_vid[i][1]
        #    f_decode_trns_mvs_per_frame.write(str(frame_type) + "|" + str(mvs) + "\n")
        
        #f_decode_trns_mvs_per_frame.close()

        with open(all_residual_f) as all_residuals:
            for line in all_residuals:
                res_for_vid.append(self.entropy_decoder_frame(line, block_size))
        
        return mv_for_vid, res_for_vid

    def decode_bitstream(self, mv_file, residual_file, intra_mode=None, intra_dur=None, block_size = None, frames=None, width=None, height=None, save_decoded_frames=True):
        if intra_dur == None: intra_dur = self.intra_dur
        if intra_mode == None: intra_mode = self.intra_mode
        if block_size == None: block_size = self.block_s
        if frames == None: frames = self.frames
        if width  == None: width  = self.w_pixels
        if height == None: height = self.h_pixels
        

        mv_per_frame,residuals_per_frame = self.decode_differential_entropy(mv_file, residual_file, block_size)

        self.mv_per_frame = mv_per_frame
        self.residuals_per_frame = residuals_per_frame

        ref_frames = [np.ones((height, width)) * 128]  # Hypothetical reference frame of all-128-values for the first frame
        decoded_frames = []
        decoded_frames = self.decode(residuals_per_frame, mv_per_frame, intra_mode, intra_dur, block_size, frames, width, height)
        
        # for i in range(frames):

        #     print("i", i)
        #     frame_type, mv_for_frame = mv_per_frame[i]

        #     if frame_type == 0:
        #        decoded_frame, _ = self.decode_frame_intra(mv_for_frame, residuals_per_frame[i], intra_mode, block_size)
        #     #    plt.imshow(decoded_frame)
        #     #    plt.show()
        #        #exit()
        #     else: 
        #        decoded_frame    = self.decode_frame_inter(ref_frames, mv_for_frame, residuals_per_frame[i], block_size)
            
        #     decoded_frames.append(decoded_frame)

        #     if i < frames - 1:  # Set the current decoded frame as the reference for the next one
        #         ref_frame = decoded_frame
        
        # if save_decoded_frames:
        #     self.decoded_vid_f = True
        #     self.decoded_vid   = decoded_frames

        return decoded_frames
    
    def save_decoded_frames(self, filename="yuv/decoded_bitstream_frames.yuv"):
        
        if self.decoded_vid_f == False:
            print("[ERROR] No decoded frames available.")
            return
        
        with open(filename, 'wb') as f:
            for data in self.decoded_vid:
                f.write(data.tobytes())
        
        if self.VBSoverlay:
            filename_overlay = filename.split(".")[0] + "overlay.yuv"
            with open(filename_overlay, "wb") as f:
                for data in self.overlay_decoded_vid:
                    f.write(data.tobytes())
