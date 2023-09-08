# Boot/BIOS Firmware Settings

## AMD CBS

### NBIO Common Options
#### SMU Common Options
##### Determinism Control: Auto
##### Determenism Slider: Auto
##### cTDP Control: Auto
##### Package Power Limit  Control: Auto
##### DF Cstates: Auto

### DF Common Options
#### Memory Addressing
##### NUMA nodes per socket: NPS1

### CPU Common Options
#### Performance
##### CCD Control: 2CCDs(BERT,RetinaNet)/Auto(ResNet50)
##### SMT Control: Disable
#### Prefetcher Settings
##### L1 Stream HW Prefetcher: Enable
##### L1 Stride Prefetcher: Auto
##### L1 Region Prefetcher: Auto
##### L2 Stream HW Prefetcher: Enable
##### L2 Up/Down Prefetcher: Auto
#### Core Performance Boost: Disable

# Management Firmware Settings

Out-of-the-box.

# Fan Settings
#### BERT and RetinaNet (5,550 RPM)

<pre>
<b>&dollar;</b> sudo ipmitool raw 0x2e 0x10 0x0a 0x3c 0 64 1 <b>50</b> 0xFF
 0a 3c 00
</pre>

#### ResNet50 (6,900 RPM)

<pre>
<b>&dollar;</b> sudo ipmitool raw 0x2e 0x10 0x0a 0x3c 0 64 1 <b>75</b> 0xFF
 0a 3c 00
</pre>

# Power Consumption Settings
### Platform Power Consumption Limit
#### ResNet50 Offline, RetinaNet Offline, BERT: Enabled (610W)
#### RetinaNet Server: Enabled (990W)
#### ResNet50 Server: Disabled

# Maximum Frequency

The maximum chip frequency is controlled through a variable called `vc`.

<pre>
BERT-99 and RetinaNet (vc = 7)
BERT-99.9 (vc = 6)
ResNet50 (vc = 10)
</pre>
