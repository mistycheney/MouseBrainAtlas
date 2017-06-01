# Labeling GUI
## Key bindings
- q: switch to 3d shift mode.
- w: switch to 3d rotate mode.
- s: switch to show score map (need to have active polygon) / histology image
- f: save structure modifications.

===============================

After downloading `CSHL_data/macros/<stack>`

upper left panel showing a picture of the slide

Use '[' and ']' to browse

Then press "Download Macros and Thumbnails"

It should begin downloading files from Gordon.


# 1. Map from Slide Position to Filename

the bottom panels show the thumbnail of three brain sections in the slide shown at the top (edited)

in the same order as they are placed on the slide

If you click on any bottom panel, then [ and ] allow you to assign this slot to another section image.

In `CSHL_data_process/MD589`,  download the following files
`MD589_sorted_filenames.txt`, `MD589_slide_position_to_fn.pkl`, `MD589_cropbox.txt`, `MD589_anchor.txt`

These are generated as a result of user interacting with the GUI, but for now we skip the interactive part and directly load these results.

Click "Load Slide Position -> Filename"

This loads the mapping for every slide, from a slide position (1,2 or 3) to a section image filename (e.g. MD589-N3-2015.07.30-16.25.24_MD589_3_0009)

A slide usually have three slots to place sections
I mark these three positions 1, 2, 3
The rightmost position is 1, middle position is 2, leftmost position is 3

`MD589_slide_position_to_fn` contains the manually corrected mapping.

If we don't have the file `MD589_slide_position_to_fn` yet, the GUI will guess the mapping
According to the rule I just mentioned: MD589-N3-2015.07.30-16.25.24_MD589_3_0009 means this is the 3rd section on slide N3
Under each bottom panel you can see the filename of it. Most of them do conform to this rule. But from time to time there are exceptions.
There are duplicates also. So I must manually inspect all of them and correct the mapping in those cases.

The image filenames usually contain this information (for example, MD589-N3-2015.07.30-16.25.24_MD589_3_0009 means this is the 3rd section on slide N3), but not always (because of data provider mistakes)

The purpose of this GUI is for the user to correct these errors if any.

Clicking "Load Slide Position -> Filename" overrides the mapping guessed by GUI with the mapping I manually corrected.

# 2. Sort the images

Ideally, the experimenter who place the sections should place them in this way.
First brain section in (N1, 1) - by this I mean the 1st position, i.e. rightmost, on slide N1
Second in (N1, 2), then (N1, 3), (IHC1, 1), (IHC1, 2), (IHC1, 3), (N2,1), (N2,2), (N2,3), (IHC2,1), (IHC2,2), (IHC2,3) ....
BUT there are exceptions to this rule because our data provider is very sloppy...

If the user does not have the file `MD589_sorted_filenames`, the GUI will guess the order according to this rule.
That is what happen when you click "Sort Sections"
Since we already have the corrected order file, you can click "Load Sorted Filenames"
This will load the corrected order.

After successfully loading the order, you should see images showing in the upper right panel

# 3. Align
# 4. Crop
# 5. Generate Masks

Loose pieces, detached tissues on slide...
