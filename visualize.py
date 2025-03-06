"""
Copyright (c) 2017, Gavin Weiguang Ding
All rights reserved.

Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle

NumDots = 4
NumConvMax = 8
NumFcMax = 20
White = 1.
Light = 0.7
Medium = 0.5
Dark = 0.3
Darker = 0.15
Black = 0.


def add_layer(patches, colors, size=(24, 24), num=5,
              top_left=[0, 0],
              loc_diff=[3, -3],
              ):
    # add a rectangle
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    for ind in range(num):
        patches.append(Rectangle(loc_start + ind * loc_diff, size[1], size[0]))
        if ind % 2:
            colors.append(Medium)
        else:
            colors.append(Light)


def add_layer_with_omission(patches, colors, size=(24, 24),
                            num=5, num_max=8,
                            num_dots=4,
                            top_left=[0, 0],
                            loc_diff=[3, -3],
                            ):
    # add a rectangle
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    this_num = min(num, num_max)
    start_omit = (this_num - num_dots) // 2
    end_omit = this_num - start_omit
    start_omit -= 1
    for ind in range(this_num):
        if (num > num_max) and (start_omit < ind < end_omit):
            omit = True
        else:
            omit = False

        if omit:
            patches.append(
                Circle(loc_start + ind * loc_diff + np.array(size) / 2, 0.5))
        else:
            patches.append(Rectangle(loc_start + ind * loc_diff,
                                     size[1], size[0]))

        if omit:
            colors.append(Black)
        elif ind % 2:
            colors.append(Medium)
        else:
            colors.append(Light)


def add_mapping(patches, colors, start_ratio, end_ratio, patch_size, ind_bgn,
                top_left_list, loc_diff_list, num_show_list, size_list):

    start_loc = top_left_list[ind_bgn] \
        + (num_show_list[ind_bgn] - 1) * np.array(loc_diff_list[ind_bgn]) \
        + np.array([start_ratio[0] * (size_list[ind_bgn][1] - patch_size[1]),
                    - start_ratio[1] * (size_list[ind_bgn][0] - patch_size[0])]
                   )




    end_loc = top_left_list[ind_bgn + 1] \
        + (num_show_list[ind_bgn + 1] - 1) * np.array(
            loc_diff_list[ind_bgn + 1]) \
        + np.array([end_ratio[0] * size_list[ind_bgn + 1][1],
                    - end_ratio[1] * size_list[ind_bgn + 1][0]])


    patches.append(Rectangle(start_loc, patch_size[1], -patch_size[0]))
    colors.append(Dark)
    patches.append(Line2D([start_loc[0], end_loc[0]],
                          [start_loc[1], end_loc[1]]))
    colors.append(Darker)
    patches.append(Line2D([start_loc[0] + patch_size[1], end_loc[0]],
                          [start_loc[1], end_loc[1]]))
    colors.append(Darker)
    patches.append(Line2D([start_loc[0], end_loc[0]],
                          [start_loc[1] - patch_size[0], end_loc[1]]))
    colors.append(Darker)
    patches.append(Line2D([start_loc[0] + patch_size[1], end_loc[0]],
                          [start_loc[1] - patch_size[0], end_loc[1]]))
    colors.append(Darker)



def label(xy, text, xy_off=[0, 4]):
    plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
             family='sans-serif', size=8)


if __name__ == '__main__':

    fc_unit_size = 2
    layer_width = 40
    flag_omit = True

    patches = []
    colors = []

    fig, ax = plt.subplots()

    ############################
    # Define the CNN architecture based on SimpleCNNClassifier
    
    # Assuming input shape of 28x28 (typical for MNIST)
    input_shape = (28, 28)
    
    # Calculate number of pooling layers (same logic as in SimpleCNNClassifier)
    h, w = input_shape
    n_pools = min(
        3,
        min(
            int(np.log2(h/2)),
            int(np.log2(w/2))
        )
    )
    
    # Define channels as in the CNN
    in_channels = 1
    channels = [16, 32, 64][:n_pools + 1]
    
    # Calculate feature map sizes after each layer
    size_list = [input_shape]
    for i in range(n_pools):
        # Size after pooling (halved in each dimension)
        new_size = (size_list[-1][0] // 2, size_list[-1][1] // 2)
        size_list.append(new_size)
    
    # Add the final conv layer (no pooling)
    size_list.append(size_list[-1])
    
    # Number of feature maps at each layer
    num_list = [in_channels] + channels
    
    # Horizontal spacing between layers
    x_diff_list = [0] + [layer_width] * (len(size_list) - 1)
    
    # Labels for each layer
    text_list = ['Inputs'] + ['Feature\nmaps'] * (len(size_list) - 1)
    
    # Vertical spacing between feature maps in each layer
    loc_diff_list = [[3, -3]] * len(size_list)

    # Calculate number of feature maps to show (limited by NumConvMax)
    num_show_list = list(map(min, num_list, [NumConvMax] * len(num_list)))
    
    # Calculate top-left position of each layer
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]

    # Print debug info
    print("Debug info:")
    print(f"n_pools: {n_pools}")
    print(f"channels: {channels}")
    print(f"size_list length: {len(size_list)}, content: {size_list}")
    print(f"num_list length: {len(num_list)}, content: {num_list}")
    
    # Draw convolutional layers
    for ind in range(len(size_list)):
        if ind < len(num_list):  # Make sure we don't go out of bounds
            if flag_omit:
                add_layer_with_omission(patches, colors, size=size_list[ind],
                                        num=num_list[ind],
                                        num_max=NumConvMax,
                                        num_dots=NumDots,
                                        top_left=top_left_list[ind],
                                        loc_diff=loc_diff_list[ind])
            else:
                add_layer(patches, colors, size=size_list[ind],
                          num=num_show_list[ind],
                          top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
            label(top_left_list[ind], text_list[ind] + '\n{}@{}x{}'.format(
                num_list[ind], size_list[ind][0], size_list[ind][1]))

    ############################
    # Draw connections between layers
    
    # For each conv block (except the last one), we have Conv+BN+ReLU+MaxPool
    # For the last conv block, we have Conv+BN+ReLU (no pooling)
    
    # Define the operations between layers
    operations = []
    for i in range(n_pools):
        operations.append('Conv+BN+ReLU+Pool')
    operations.append('Conv+BN+ReLU')
    
    # Define connection parameters
    start_ratio_list = [[0.4, 0.5]] * len(operations)
    end_ratio_list = [[0.4, 0.5]] * len(operations)
    patch_size_list = [(3, 3)] * len(operations)  # All kernels are 3x3
    
    # Draw connections - make sure we don't go out of bounds
    for ind in range(len(operations)):
        if ind < len(size_list) - 1:  # Make sure we have a next layer to connect to
            # Make sure num_show_list has enough elements
            while len(num_show_list) <= ind + 1:
                num_show_list.append(1)  # Add a default value
                
            add_mapping(
                patches, colors, start_ratio_list[ind], end_ratio_list[ind],
                patch_size_list[ind], ind,
                top_left_list, loc_diff_list, num_show_list, size_list)
            label(top_left_list[ind], operations[ind] + '\n{}x{} kernel'.format(
                patch_size_list[ind][0], patch_size_list[ind][1]), xy_off=[26, -65]
            )

    ############################
    # Calculate flattened size (approximate based on last feature map size)
    last_conv_size = size_list[-1]
    last_conv_channels = channels[-1]
    flat_size = last_conv_size[0] * last_conv_size[1] * last_conv_channels
    
    # Define hidden layer sizes as in the CNN
    hidden_size = min(256, flat_size)
    hidden_size_2 = hidden_size // 4
    output_size = 2
    
    # Fully connected layers
    fc_size_list = [(fc_unit_size, fc_unit_size)] * 3
    fc_num_list = [flat_size, hidden_size, output_size]
    fc_num_show_list = list(map(min, fc_num_list, [NumFcMax] * len(fc_num_list)))
    
    # Calculate positions
    fc_x_diff_list = [sum(x_diff_list) + layer_width, layer_width, layer_width]
    fc_top_left_list = np.c_[np.cumsum(fc_x_diff_list), np.zeros(len(fc_x_diff_list))]
    fc_loc_diff_list = [[fc_unit_size, -fc_unit_size]] * len(fc_top_left_list)
    fc_text_list = ['Hidden\nunits', 'Hidden\nunits', 'Outputs']

    # Draw fully connected layers
    for ind in range(len(fc_size_list)):
        if flag_omit:
            add_layer_with_omission(patches, colors, size=fc_size_list[ind],
                                    num=fc_num_list[ind],
                                    num_max=NumFcMax,
                                    num_dots=NumDots,
                                    top_left=fc_top_left_list[ind],
                                    loc_diff=fc_loc_diff_list[ind])
        else:
            add_layer(patches, colors, size=fc_size_list[ind],
                      num=fc_num_show_list[ind],
                      top_left=fc_top_left_list[ind],
                      loc_diff=fc_loc_diff_list[ind])
        label(fc_top_left_list[ind], fc_text_list[ind] + '\n{}'.format(
            fc_num_list[ind]))

    # Layer operation labels
    fc_op_text_list = ['Flatten', 'FC+ReLU+Dropout', 'FC+ReLU+Dropout']
    for ind in range(len(fc_size_list)):
        label(fc_top_left_list[ind], fc_op_text_list[ind], xy_off=[-10, -65])

    ############################
    # Apply colors and add to plot
    for patch, color in zip(patches, colors):
        patch.set_color(color * np.ones(3))
        if isinstance(patch, Line2D):
            ax.add_line(patch)
        else:
            patch.set_edgecolor(Black * np.ones(3))
            ax.add_patch(patch)

    plt.tight_layout()
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    fig.set_size_inches(10, 3)  # Slightly larger to fit all details

    fig_dir = './'
    fig_ext = '.png'
    fig.savefig(os.path.join(fig_dir, 'cnn_architecture_fig' + fig_ext),
                bbox_inches='tight', pad_inches=0)