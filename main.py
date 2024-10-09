import time
import os, shutil
import numpy as np
import decoder as dec
import Y_video_codec as codec
import matplotlib.pyplot as plt
import video_manager as v_manager

class run:

    def __init__(self,ParallelMode=0):
        self.ParallelMode=ParallelMode
# Encode/Decoder config #######################################################
    def main(self):
        block_size   = 16
        search_range = 16
        Qp           = 5
        intra_dur    = 10
        intra_mode   = 0
        frames       = 21
        h_pixels     = 288
        w_pixels     = 352
        nRefFrames   = 1
        FMEEnable    = True 
        fast_me      = True 
        VBSEnable    = False 
        VBSoverlay   = False 
        lam          = 0.015
        debug_prints = True
        RCFlag       = 0
        targetBR     = "500000 kbps" # Format: "<num><space><units>" Make sure there is a space before the units. Units can be: bps, kbps, or mbps
        intra_thresh = 25000
        ParallelMode = self.ParallelMode
        mv_file      = f"files/mvs_per_frame_{self.ParallelMode}.txt"
        residual_file= f"files/res_per_frame_{self.ParallelMode}.txt"
        size_arr_intra = [17464.75, 16530.125, 13534.5, 10611.75, 7970.375, 5716.0, 3912.0, 2505.0, 1546.75, 988.25, 770.125, 703.25]
        size_arr_inter = [14635.25, 11306.25, 8614.375, 6250.25, 4305.5, 3093.25, 2169.25, 1537.125, 939.25, 604.5, 520.875, 498.75]
        qp_tables = [size_arr_intra, size_arr_inter]
        
        # Video Manager ###############################################################
        vm = v_manager.Video_Manager("video/cif.yuv", 288, 352, frames, "yuv_420")
        vm.upscale_yuv420_to_yuv444()
        vm.convert_yuv444_to_rgb()
        Y_ONLY = vm.extract_y_only()
        ###############################################################################

        if debug_prints: print("[INFO] YUV 4:2:0 file read and converted. Now running encoder.")

        # Encoder #####################################################################
        encoder = codec.Y_Video_codec(vm.h_pixels, vm.w_pixels, vm.frames, block_size, search_range, Qp, intra_dur, intra_mode, lam, VBSEnable, nRefFrames=nRefFrames, y_only_frame_arr=Y_ONLY, fast_me=fast_me, FMEEnable=FMEEnable, RCFlag=RCFlag, targetBR=targetBR, frame_rate=30, qp_rate_tables=qp_tables, intra_thresh=intra_thresh, ParallelMode=ParallelMode)
        if debug_prints: print("[INFO] Encoding")
        psnr = encoder.encode(block_size=block_size)
        if debug_prints: print("[INFO] Done")
        if debug_prints: print("[INFO] Generating Bitstream")
        encoder.transmit_bitstream(block_size=block_size, mv_file=mv_file, residual_file=residual_file)
        if debug_prints: print("[INFO] Done")
        return psnr
        ###############################################################################

        # Decoder #####################################################################
        # if debug_prints: print("[INFO] Decoding Bitstreaming")
        # decoder = dec.decoder(intra_mode, intra_dur, block_size, vm.frames, vm.h_pixels, vm.w_pixels, Qp, nRefFrames, FMEEnable, lam, VBSEnable, VBSoverlay, RCFlag=RCFlag, targetBR=targetBR, frame_rate=30, qp_rate_tables=qp_tables,ParallelMode=ParallelMode)
        # decoder.decode_bitstream(mv_file, residual_file, block_size=block_size)
        # if debug_prints: print("[INFO] Done")
        # if debug_prints: print("[INFO] Saving decoded frames")
        # decoder.save_decoded_frames()
        # if debug_prints: print("[INFO] Done")
        ###############################################################################
    ###############################################################################

    # To empty folder contents before each run
    def delete_folder_contents(self,folders=['files','yuv']):
        for folder in folders:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Generate Qp Tables for Rate control algorithm
    def generate_Qp_tables(self,total_num_frames, I_period_list, qp_list, preserve_logs=False):
        #deleting folder contents before run
        self.delete_folder_contents()
        size_arr_intra=[]
        size_arr_inter=[]
        preserve_logs = False

        for i_dur in I_period_list:
            for qp in qp_list:
                Qp = qp
                intra_dur = i_dur
                run(debug_prints=False, Qp=qp, intra_dur=i_dur)

                mvs_file=f'files/mvs_per_frame.txt'
                res_file=f'files/res_per_frame.txt'

                # Segregate Inter and Intra frams into different buckets to calculate size
                mvs_file_intra_f=f'files/temp_mvs_per_frame_Qp_{qp}_I_{i_dur}_intra.txt'
                res_file_intra_f=f'files/temp_res_per_frame_Qp_{qp}_I_{i_dur}_intra.txt'
                mvs_file_inter_f=f'files/temp_mvs_per_frame_Qp_{qp}_I_{i_dur}_inter.txt'
                res_file_inter_f=f'files/temp_res_per_frame_Qp_{qp}_I_{i_dur}_inter.txt'

                with open(mvs_file, "r") as mvs_f, open(mvs_file_intra_f, "w") as temp_mvs_intra_f, open(mvs_file_inter_f, "w") as temp_mvs_inter_f:
                    for l_num, line in enumerate(mvs_f):
                        if l_num % i_dur == 0:
                            temp_mvs_intra_f.write(line)
                        else:
                            temp_mvs_inter_f.write(line)

                with open(res_file, "r") as mvs_f, open(res_file_intra_f, "w") as temp_res_intra_f, open(res_file_inter_f, "w") as temp_res_inter_f:
                    for l_num, line in enumerate(mvs_f):
                        if l_num % i_dur == 0:
                            temp_res_intra_f.write(line)
                        else:
                            temp_res_inter_f.write(line)

                intra_file_size_mvs = os.path.getsize(mvs_file_intra_f)
                intra_file_size_res = os.path.getsize(res_file_intra_f)
                inter_file_size_mvs = os.path.getsize(mvs_file_inter_f)
                inter_file_size_res = os.path.getsize(res_file_inter_f)

                total_size_intra = (intra_file_size_mvs+intra_file_size_res) * 8
                total_size_inter = (inter_file_size_mvs+inter_file_size_res) * 8

                num_intra_frames = total_num_frames//i_dur
                num_inter_frames = total_num_frames - num_intra_frames

                if num_intra_frames == 0: size_per_row_intra = 0
                else: size_per_row_intra = (total_size_intra/num_intra_frames)//(288/16)

                if num_inter_frames == 0: size_per_row_inter = 0
                else: size_per_row_inter = (total_size_inter/num_inter_frames)//(288/16)

                # print(f'Qp: {qp} | i_dur: {i_dur} :: intra_size_per_row: {size_per_row_intra}, inter_size_per_row: {size_per_row_inter}')
                size_arr_intra.append(size_per_row_intra)
                size_arr_inter.append(size_per_row_inter)

                # Delete the excess files, clean up the workspace
                mvs_per_frame_raw=f'files/mvs_per_frame_raw.txt'
                y_only_decoded_f =f'yuv/y_only_decoded.yuv'
                y_only_reconstructed_f = f'yuv/y_only_reconstructed.yuv'
                decoded_bitstream_frames_f = "yuv/decoded_bitstream_frames.yuv"
                decoded_bitstream_frames_overlay_f = f"yuv/decoded_bitstream_framesoverlay.yuv"
                
                if preserve_logs == False: 
                    os.remove(mvs_file_inter_f)
                    os.remove(mvs_file_intra_f)
                    os.remove(res_file_inter_f)
                    os.remove(res_file_intra_f)
                    os.remove(mvs_file)
                    os.remove(res_file)
                    os.remove(mvs_per_frame_raw)
                    os.remove(y_only_decoded_f)
                    os.remove(y_only_reconstructed_f)
                    os.remove(decoded_bitstream_frames_f)
                    os.remove(decoded_bitstream_frames_overlay_f)

        # print("Qp Table for intra frame: ")
        # size_arr_intra.reverse()
        # print(size_arr_intra)
        # print("Qp Table for inter frame: ")
        # size_arr_inter.reverse()
        # print(size_arr_inter)

        return [size_arr_intra, size_arr_inter]

    # Run an instance of the entire system, File extraction -> Encoder -> Decoder
    def plot_psnr_per_bit(self, value_json):
        print(value_json)
        plt.figure(figsize=(14, 8))
        # plt.title(f"Average PSNR vs total bits: {filename} block_size: {block} x {block}")
        plt.title(f"RD Plot of CIF for different RCflag values")
        plt.ticklabel_format(useOffset=False)
        plt.xlabel("Total bits")
        plt.ylabel("Average PSNR")

        total_bits_arr ={}
        avg_psnr_arr ={}
        legend_arr = []
        for value in value_json:
            # print(value)
            for key in value.keys():
                annotation = key.split(',')[0]
                legend = key.split(',')[1]
                # legend=key

                if legend in total_bits_arr:
                    total_bits_arr[legend].append(value[key]['total_bits'])
                else: 
                    total_bits_arr[legend] = [value[key]['total_bits']]

                if legend in avg_psnr_arr:
                    avg_psnr_arr[legend].append(value[key]['avg_psnr'])
                else: 
                    avg_psnr_arr[legend] =[value[key]['avg_psnr']]
                # avg_psnr_arr[legend.replace("=", "_")].append(value[key]['avg_psnr'])
                plt.annotate(annotation, # this is the text
                 (value[key]['total_bits'], value[key]['avg_psnr']), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
                # print(value[key]['total_bits'])
                # print(value[key]['avg_psnr'])
        
        for key in total_bits_arr.keys():
            legend_arr.append(key)
            plt.plot(total_bits_arr[key], avg_psnr_arr[key], marker='*')

        print(total_bits_arr)
        print(avg_psnr_arr)
        print(legend_arr)
        plt.grid(True)
        plt.legend(legend_arr, loc='upper left')    
        plt.show()
#time_arr=[307.2293894290924, 109.63216543197632, 34.349550008773804]a

    def plot_time(self, time_arr_latest):
        # time_arr_latest = [10.607625961303711, 316.915593624115, 105.88217234611511, 34.789713621139526]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(time_arr_latest)), time_arr_latest, marker='o', color='green')  # Set color to green

        # Adding labels and title
        plt.xlabel('Parallelism Type')
        plt.ylabel('Encoding Time (seconds)')
        plt.title('Comparison of Encoding Times for Different Types of Parallelism')

        # Setting x-axis ticks and labels
        plt.xticks(range(len(time_arr_latest)), ['Type 0', 'Type 1', 'Type 2', 'Type 3'])

        # Adding encoding time labels to data points
        for i, time in enumerate(time_arr_latest):
            plt.annotate(f'{time:.2f} s', (i, time), textcoords="offset points", xytext=(0, 5), ha='center')

        # Set custom axis ranges for better visualization
        plt.ylim(0, max(time_arr_latest) + 20)

        # Show the plot
        # plt.grid(True)
        plt.show()


if __name__ == '__main__': 
    
    # Qp and Intra frame duration conrners to explore
    

    #qp_tables = generate_Qp_tables(frames,I_period_list,qp_list)
    # print(f"Running for Qp={Qp}, Intra Frame Duration={intra_dur}, and RCFlag enabled")
    # For time arr
    time_arr=[]
    ParallelMode_list=[0,1,2,3]
    for i in ParallelMode_list:
        start_time=time.time()
        obj=run(i)
        obj.main()
        end_time=(time.time()-start_time)
        time_arr.append(end_time)
        print(end_time)
    #time with pool: 169.93
    print('Time taken by each mode respectively: ',time_arr)
    obj=run(0)
    # time_arr = [10.607625961303711, 316.915593624115, 105.88217234611511, 34.789713621139526]

    obj.plot_time(time_arr)
