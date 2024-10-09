import numpy as np
import matplotlib.pyplot as plt
import video_manager as v_manager
import Y_video_codec as codec
import decoder as dec
import os, shutil
import time

class main:
    def __init__(self, targetBR, idx,qp,RCflag):
        self.targetBR = targetBR
        self.idx = idx
        self.Qp=qp
        self.RCflag = RCflag

    # Run an instance of the entire system, File extraction -> Encoder -> Decoder
    def main(self,Qp=5,debug_prints=False, qp_tables=None):
        start_time=time.time()
        block_size   = 16
        search_range = 16
        Qp           = self.Qp
        intra_dur    = 21
        intra_mode   = 0
        frames       = 21
        h_pixels     = 288
        w_pixels     = 352
        nRefFrames   = 1
        FMEEnable    = True 
        fast_me      = True 
        VBSEnable    = True 
        VBSoverlay   = False 
        lam          = 0.015
        debug_prints = True
        RCFlag       = self.RCflag
        # Sample Usage
        # targetBR     = "2516582 bps" 
        # Format: "<num><space><units>" Make sure there is a space before the units. Units can be: bps, kbps, or mbps
        targetBR = self.targetBR
        intra_thresh = 70000
        dQPLimit     = 2
        mv_file      = f"files/mvs_per_frame_{self.idx}.txt"
        residual_file= f"files/res_per_frame_{self.idx}.txt"
        qp_tables = [size_arr_intra, size_arr_inter]
        
        # Video Manager ###############################################################
        vm = v_manager.Video_Manager("video/cif.yuv", 288, 352, frames, "yuv_420")
        vm.upscale_yuv420_to_yuv444()
        vm.convert_yuv444_to_rgb()
        Y_ONLY = vm.extract_y_only()
        ###############################################################################

        if debug_prints: print("[INFO] YUV 4:2:0 file read and converted. Now running encoder.")

        # Encoder #####################################################################
        encoder = codec.Y_Video_codec(vm.h_pixels, vm.w_pixels, vm.frames, block_size, search_range, Qp, intra_dur, intra_mode, lam, VBSEnable, nRefFrames=nRefFrames, y_only_frame_arr=Y_ONLY, fast_me=fast_me, FMEEnable=FMEEnable, RCFlag=RCFlag, targetBR=targetBR, frame_rate=30, qp_rate_tables=qp_tables, intra_thresh=intra_thresh)
        if debug_prints: print("[INFO] Encoding")
        psnr = encoder.encode(block_size=block_size)
        if debug_prints: print("[INFO] Done")
        if debug_prints: print("[INFO] Generating Bitstream")
        encoder.transmit_bitstream(block_size=block_size, mv_file=mv_file, residual_file=residual_file)
        if debug_prints: print("[INFO] Done")
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n',time.time()-start_time,'\n')
        ###############################################################################

        # Decoder #####################################################################
        if debug_prints: print("[INFO] Decoding Bitstreaming")
        decoder = dec.decoder(intra_mode, intra_dur, block_size, vm.frames, vm.h_pixels, vm.w_pixels, Qp, nRefFrames, FMEEnable, lam, VBSEnable, VBSoverlay, RCFlag=RCFlag, targetBR=targetBR, frame_rate=30, qp_rate_tables=qp_tables)
        decoder.decode_bitstream(mv_file, residual_file, block_size=block_size)
        if debug_prints: print("[INFO] Done")
        if debug_prints: print("[INFO] Saving decoded frames")
        decoder.save_decoded_frames()
        if debug_prints: print("[INFO] Done")
        ###############################################################################
