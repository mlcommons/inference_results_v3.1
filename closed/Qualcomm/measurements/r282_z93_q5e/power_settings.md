# Boot/BIOS Firmware Settings

## AMD CBS

### NBIO Common Options
#### SMU Common Options
##### Determinism Control: Manual
##### Determinism Slider: Performance
##### cTDP: Auto
##### DF Cstates: Auto

### DF Common Options

#### Scrubber
##### DRAM scrub time: Auto
##### Poisson scrubber control: Auto
##### Redirect scrubber control: Auto

#### Memory Addressing
##### NUMA nodes per socket: NPS4

### CPU Common Options
#### Performance
##### SMT Control: Disable

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

# Maximum Frequency

The maximum chip frequency is controlled through a variable called `vc`.

<pre>
BERT (vc = 8)
ResNet50 and RetinaNet (vc = 10)
</pre>
