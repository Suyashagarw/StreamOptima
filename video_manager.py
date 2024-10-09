import numpy as np
import matplotlib.pyplot as plt

class Video_Manager:
   
    def __init__(self, raw_f, h_pixels, w_pixels, frames, v_type):
        self.current_f           = raw_f
        self.h_pixels            = h_pixels
        self.w_pixels            = w_pixels
        self.frames              = frames
        
        self.v_yuv420            = False
        self.v_yuv444            = False
        self.v_rgb               = False

        self.vid_frames_yuv420   = None
        self.vid_frames_yuv444   = None
        self.vid_frames_rgb      = None

        if v_type == "yuv_420":
            self.v_yuv420               = True
            self.num_y_p_yuv420         = int(h_pixels * w_pixels)
            self.num_u_p_yuv420         = int(self.num_y_p_yuv420/4)
            self.num_v_p_yuv420         = self.num_u_p_yuv420
            self.frame_size_p           = self.num_y_p_yuv420 + self.num_u_p_yuv420 + self.num_v_p_yuv420
            self.vid_frames_yuv420      = self.raw_yuv420_to_frame_arr(raw_f, h_pixels, w_pixels)
        elif v_type == "yuv_444":
            self.num_y_p_yuv_444    = int(h_pixels * w_pixels)
            self.num_u_p_yuv_444    = self.num_y_p_yuv_444
            self.num_v_p_yuv_444    = self.num_u_p_yuv_444
            self.frame_size_p       = self.num_y_p_yuv_444 + self.num_u_p_yuv_444 + self.num_v_p_yuv_444
            self.v_yuv444           = False
            self.vid_frames_yuv444  = self.raw_yuv444_to_frame_arr(raw_f, h_pixels, w_pixels)
        elif v_type == "rgb":
            print("[ERROR] Cannot parse RGB video file!")
            #self.num_r_p_rgb        = int(h_pixels * w_pixels)
            #self.num_g_p_rgb        = self.num_r_p_rgb
            #self.num_b_p_rgb        = self.num_g_p_rgb
            #self.frame_size_p       = self.num_r_p_rgb + self.num_g_p_rgb + self.num_b_p_rgb
            #self.v_rgb              = False
            #self.vid_frames_rgb     = self.raw_rgb_to_frame_arr(raw_f, h_pixels, w_pixels, frames)
                

    def print_status(self):
        print("################################################")
        print("Video Manager status")
        print("################################################")
        print("\tCurrent Video File  : ",self.current_f) 
        print("\tVideo Height (in px): ",self.h_pixels)
        print("\tVideo Width (in px) : ",self.w_pixels)
        print("\tVideo # frames      : ",self.frames)
        
        print("\tYUV 4:2:0 available : ",self.v_yuv420)
        print("\tYUV 4:4:4 available : ",self.v_yuv444)
        print("\tRGB available       : ",self.v_rgb)
        print("################################################")

    # Converts raw yuv file to a numpy array of pixels
    # h_pixel: Height of a frame in pixels
    # w_pixel: Width of a frame in pixels
    # frames : Number of frames in the video
    #         0                    1                    2
    # *---------------*   *---------------*    *---------------*
    # |   Y   | U | V |   |   Y   | U | V |    |   Y   | U | V |
    # *---------------*   *---------------*    *---------------*
    @staticmethod
    def raw_yuv420_to_frame_arr (raw_yuv, h_pixel, w_pixel, frames=21, v_file=True):
        '''Converts raw 4:2:0 yuv file to a numpy array of pixels'''
   
        if v_file:
            raw_vid_arr = np.fromfile(raw_yuv, dtype='uint8')
        else:
            raw_vid_arr = raw_yuv

        frame_size_p = int(h_pixel * w_pixel * 1.5)
       
        if frames == None:
            frames = raw_vid_arr.shape[0]/frame_size_p
   
        vid_frames = raw_vid_arr.reshape(frames, frame_size_p)
   
        return vid_frames
   
    # Converts raw yuv file to a numpy array of pixels
    # h_pixel: Height of a frame in pixels
    # w_pixel: Width of a frame in pixels
    # frames : Number of frames in the video
    @staticmethod
    def raw_yuv444_to_frame_arr (raw_yuv, h_pixel, w_pixel, frames=300, v_file=True):
        if v_file:
            raw_vid_arr = np.fromfile(raw_yuv, dtype='uint8')
        else:
            raw_vid_arr = raw_yuv
   
        frame_size_p = h_pixel * w_pixel

        if frames == None:
            frames = raw_vid_arr.shape[0]/frame_size_p
   
        vid_frames = raw_vid_arr.reshape(frames, 3, h_pixel, w_pixel) # Index: Frame, component (Y/U/V), h_pixel, w_pixel
   
        return vid_frames

    def view_frame_yuv420(self, frame, selector=0):
        if selector == 0:   # Only Y
            frame_v = self.vid_frames_yuv420[frame][0:self.num_y_p_yuv420].reshape(self.h_pixels, self.w_pixels)
            plt.imshow(frame_v, cmap = 'gray')
            plt.show()
        elif selector == 1: # Only U
            frame_v = self.vid_frames_yuv420[frame][self.num_y_p_yuv420:self.num_y_p_yuv420+self.num_u_p_yuv420].reshape(int(self.h_pixels/2), int(self.w_pixels/2))
            plt.imshow(frame_v, cmap = 'gray')
            plt.show()
        elif selector == 2: # Only V
            frame_v = self.vid_frames_yuv420[frame][self.num_y_p_yuv420+self.num_u_p_yuv420:self.num_y_p_yuv420+self.num_u_p_yuv420+self.num_v_p_yuv420].reshape(int(self.h_pixels/2), int(self.w_pixels/2))
            plt.imshow(frame_v, cmap = 'gray')
            plt.show()

    def view_frame_yuv444(self, frame, selector=0):
        frame_v = self.vid_frames_yuv444[frame][selector]
        plt.imshow(frame_v, cmap = 'gray')
        plt.show()

    def view_frame_rgb(self, frame, selector=0):
        if selector == 3:  # Show frame with all components RGB
            frame_v = self.vid_frames_rgb[frame]
        else: # 0: R, 1: G, 2: B
            frame_v = self.vid_frames_rgb[frame, :, :, selector]
        
        plt.imshow(frame_v)
        plt.show()
    
    def view_frame(self, v_file, frame, selector=0):
        if v_file == 0: # Display YUV 420
            if self.v_yuv420 == True:
                self.view_frame_yuv420(frame, selector)
            else:
                print("[ERROR] No YUV 420 available!")
        elif v_file == 1: # Display YUV 444
            if self.v_yuv444 == True:
                self.view_frame_yuv444(frame, selector)
            else:
                print("[ERROR] No YUV 444 available!")
        elif v_file == 2: # Diplay RGB
            if self.v_rgb == True:
                self.view_frame_rgb(frame, selector)
            else:
                print("[ERROR] No RGB available!")

    def upscale_yuv420_to_yuv444(self, replace=True):
        if self.v_yuv420 is False:
            print("[ERROR] No YUV 4:2:0 file available to convert!");
            return None

        converted_vid_arr = None

        for i in range(0, self.frames):
            YUV_420   = self.vid_frames_yuv420[i]
            YUV_444_Y = YUV_420[0:self.num_y_p_yuv420]
           
            YUV_444_U = YUV_420[self.num_y_p_yuv420:self.num_y_p_yuv420+self.num_u_p_yuv420].reshape(int(self.h_pixels/2), int(self.w_pixels/2))
            YUV_444_U = YUV_444_U.repeat(2, 0)
            YUV_444_U = YUV_444_U.repeat(2, 1)
            YUV_444_U = YUV_444_U.reshape(self.h_pixels*self.w_pixels,)

            YUV_444_V = YUV_420[self.num_y_p_yuv420+self.num_u_p_yuv420:self.num_y_p_yuv420+self.num_u_p_yuv420+self.num_v_p_yuv420].reshape(int(self.h_pixels/2), int(self.w_pixels/2))
            YUV_444_V = YUV_444_V.repeat(2, 0)
            YUV_444_V = YUV_444_V.repeat(2, 1)
            YUV_444_V = YUV_444_V.reshape(self.h_pixels*self.w_pixels,)
           
            if i == 0:
                converted_vid_arr = np.hstack((YUV_444_Y, YUV_444_U, YUV_444_V))
            else:
                converted_vid_arr = np.hstack((converted_vid_arr, YUV_444_Y, YUV_444_U, YUV_444_V))
       
        if replace:
            self.v_yuv444 = True
            self.vid_frames_yuv444 = self.raw_yuv444_to_frame_arr(converted_vid_arr, self.h_pixels, self.w_pixels, self.frames, False)
            self.num_y_p_yuv_444 = self.num_y_p_yuv420
            self.num_u_p_yuv_444 = self.num_y_p_yuv_444
            self.num_v_p_yuv_444 = self.num_y_p_yuv_444
       
        return converted_vid_arr

    def convert_yuv444_to_rgb(self, replace=True):
        if self.v_yuv444 is False:
            print("[ERROR] No YUV 4:4:4 file available to convert!");
            return None
       
        converted_vid_arr = None
       
        conv_mat = np.array([
            [1.164,  0.000,  2.018], [1.164, -0.813, -0.391],[1.164,  1.596,  0.000]
            ])

        for i in range(0, self.frames):
            YUV_444 = self.vid_frames_yuv444[i]
            YUV_444_Y = YUV_444[0].reshape(self.h_pixels, self.w_pixels, 1)
            YUV_444_U = YUV_444[1].reshape(self.h_pixels, self.w_pixels, 1)
            YUV_444_V = YUV_444[2].reshape(self.h_pixels, self.w_pixels, 1)

            YUV_444 = np.concatenate((YUV_444_Y, YUV_444_U, YUV_444_V), axis=2).astype(np.float32)
            YUV_444[:, :,  0] = YUV_444[:, :, 0].clip(16, 235) - 16
            YUV_444[:, :, 1:] = YUV_444[:, :, 1:].clip(16, 240) - 128

            RGB = np.matmul(YUV_444, conv_mat.T).clip(0, 255).astype("uint8")
            RGB = RGB.reshape(1,RGB.shape[0], RGB.shape[1], RGB.shape[2])


            if i == 0:
                converted_vid_arr = RGB
            else:
                converted_vid_arr = np.vstack((converted_vid_arr, RGB))
           
        if replace:
            self.v_rgb = True
            #self.vid_frames_rgb = self.raw_yuv444_to_frame_arr(converted_vid_arr, self.h_pixels, self.w_pixels, self.frames, False)
            self.vid_frames_rgb = converted_vid_arr # Index: Frame, h_pixel, w_pixel, component (R/G/B)
            self.num_r_p_rgb = self.num_y_p_yuv_444
            self.num_g_p_rgb = self.num_u_p_yuv_444
            self.num_b_p_rgb = self.num_v_p_yuv_444

        return converted_vid_arr
    
    # TODO: This function doesn't really work
    def add_noise(self, selector, scale=20):
        if not (selector == "YUV_420" and self.v_yuv420 == False):
            self.vid_frames_yuv420 = np.random.normal(self.vid_frames_yuv420, scale)
        elif not (selector == "YUV_444" and self.v_yuv444 == False):
            self.vid_frames_yuv444 = np.random.normal(self.vid_frames_yuv444, scale)
        elif not (selector == "RGB" and self.v_rgb == False):
            self.vid_frames_rgb = np.random.normal(self.vid_frames_rgb, scale)
        else:
            print("[ERROR] Cannot add random noise, requested file type not available")

    def extract_y_only(self, dump=True):
        Y_ONLY_YUV = None
        if self.v_yuv444 == False:
            print("[ERROR] No YUV 4:4:4 file avialable. Currenlty, tool can only extract Y-Only files from YUV 4:4:4.")
        else:
            Y_ONLY_YUV = self.vid_frames_yuv444[:,0,:,:]
            
        return Y_ONLY_YUV

    def save_y_only(self, filename, y_data_list):
        with open(filename, 'wb') as f:
            for idx, data in enumerate(y_data_list):
                f.write(data.tobytes())
#dummy = Video_Manager("video/akiyo_cif.yuv", 288, 352, 300, "yuv_420")
#dummy.upscale_yuv420_to_yuv444()
#dummy.convert_yuv444_to_rgb()
#Y_ONLY = dummy.extract_y_only()