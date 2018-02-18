from tkinter import *
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image
from PIL import ImageTk
import imutils
import numpy as np
#import tkFileDialog
import cv2
import os

from grab_flow_chart import *
from heuristics import *


def resize_to_square(image, side_length):
    old_size = image.size
    ratio =  side_length / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    image = image.resize(new_size, Image.ANTIALIAS)
    temp_image = Image.new("RGBA", (side_length, side_length))
    temp_image.paste(image, ((side_length - new_size[0]) // 2, (side_length - new_size[1]) // 2))
    
    return temp_image

class ImageGui:

    def __init__(self):

        self.sm_Image = 245 #230
        self.lg_Image = 800 #750

        this_path = os.path.dirname(os.path.realpath(__file__))
        assets = os.path.join(this_path, "assets")
        icon = os.path.join(assets, "icon.ico")

        self.root = Tk()
        self.root.title("Flow chart image processor")
        self.root.iconbitmap(icon)
        
        self.IMAGEPATH = None
        self.IMAGEPROCESSOR_OBJECT = None

        self.defaults = {
            'threshLevel': 115,
            'boxApproxParameter': 0.02,
            'sizeThreshold': 0.2,
            'lineDilateIterations': 3,
            'equalizeBackground': True,
            'skipClosing': False,
            'maskThickness': 8,
            'unstack': False,
            'directional': False,
            'prefer_linked': False,
            'boxDilationIterations': 1,
            'skipDilation': False,
            'duplicateThreshold': 10
        }

        self.threshLevel = self.defaults['threshLevel']
        self.boxApproxParameter = self.defaults['boxApproxParameter']
        self.sizeThreshold = self.defaults['sizeThreshold']
        self.lineDilateIterations = self.defaults['lineDilateIterations']

        self.equalizeBackground = self.defaults['equalizeBackground']
        self.equalizeBackgroundVar = BooleanVar()
        
        self.skipClosing = self.defaults['skipClosing']
        self.skipClosingVar = BooleanVar()
        
        self.maskThickness = self.defaults['maskThickness']

        self.unstack = self.defaults['unstack']
        self.unstackVar = BooleanVar()

        self.directional = self.defaults['directional']
        self.directionalVar = BooleanVar()

        self.prefer_linked = self.defaults['prefer_linked']
        self.prefer_linkedVar = BooleanVar()

        self.boxDilationIterations = self.defaults['boxDilationIterations']

        self.skipDilation = self.defaults['skipDilation']
        self.skipDilationVar = BooleanVar()

        self.duplicateThreshold = self.defaults['duplicateThreshold']

        blank = np.full((self.lg_Image, self.lg_Image), 240, dtype=np.uint8)
        blank = Image.fromarray(blank)
        blank_sm = blank.resize((self.sm_Image, self.sm_Image))
        blank_lg = blank.resize((self.lg_Image, self.lg_Image))
        
        blank_sm = ImageTk.PhotoImage(blank_sm)
        blank_lg = ImageTk.PhotoImage(blank_lg)
        
        self.left_frame = Frame(self.root, width=750, height=750)
        self.left_frame.pack(side="left")

        self.middle_frame = Frame(self.root, width=750, height=750)
        self.middle_frame.pack(side="left")

        self.right_frame = Frame(self.root, width=250, height=750)
        self.right_frame.pack(side="left")

        Label(self.left_frame, text="Original image").grid(column=0, row=0, sticky=(N))
        self.panelA1 = Label(self.left_frame, image = blank_sm, width = self.sm_Image,  relief="groove")
        self.panelA1.image = blank_sm
        self.panelA1.grid(column=0, row=1, sticky=(N, W), padx=10)

        Label(self.left_frame, text="Thresholded image").grid(column=1, row=0, sticky=(N))
        self.panelA2 = Label(self.left_frame, image = blank_sm, width = self.sm_Image,  relief="groove")
        self.panelA2.image = blank_sm
        self.panelA2.grid(column=1, row=1, sticky=(N, E), padx=10)

        Label(self.left_frame, text="Dilated/Closed image").grid(column=0, row=2, sticky=(N))
        self.panelA3 = Label(self.left_frame, image = blank_sm, width = self.sm_Image,  relief="groove")
        self.panelA3.image = blank_sm
        self.panelA3.grid(column=0, row=3, sticky=(W), padx=10)

        Label(self.left_frame, text="Contours found (boxes in blue)").grid(column=1, row=2, sticky=(N))
        self.panelA4 = Label(self.left_frame, image = blank_sm, width = self.sm_Image,  relief="groove")
        self.panelA4.image = blank_sm
        self.panelA4.grid(column=1, row=3, sticky=(E), padx=10)

        Label(self.left_frame, text="Boxes kept").grid(column=0, row=4, sticky=(N))
        self.panelA5 = Label(self.left_frame, image = blank_sm, width = self.sm_Image,  relief="groove")
        self.panelA5.image = blank_sm
        self.panelA5.grid(column=0, row=5, sticky=(S, W), padx=10)

        Label(self.left_frame, text="Lines").grid(column=1, row=4, sticky=(N))
        self.panelA6 = Label(self.left_frame, image = blank_sm, width = self.sm_Image,  relief="groove")
        self.panelA6.image = blank_sm
        self.panelA6.grid(column=1, row=5, sticky=(S, E), padx=10)

        Label(self.middle_frame, text="Result").pack()
        self.panelB = Label(self.middle_frame, image=blank_lg, width=self.lg_Image,  relief="groove")
        self.panelB.image = blank_lg
        self.panelB.pack(side="left", padx=10)

        btn = Button(self.right_frame, text="Select an image", command=self.select_image)
        btn.pack(side="top", fill="both", expand="yes", padx="10", pady="10")

        self.reset_btn = Button(self.right_frame, text="Reset defaults", command=self.reset_defaults, state=DISABLED)
        self.reset_btn.pack(side="top", fill="both", expand="yes", padx="10", pady="10")

        threshLabel = Label(self.right_frame, text="Threshold level")
        threshLabel.pack()
        self.threshSlider = Scale(self.right_frame, from_=0, to=255, orient=HORIZONTAL, tickinterval=255, length=250, command=self.threshChange, state=DISABLED)
        self.threshSlider.set(self.threshLevel)
        self.threshSlider.pack()

        boxDilationIterationsLabel = Label(self.right_frame, text="Box dilation iterations")
        boxDilationIterationsLabel.pack()
        self.boxDilationIterationsSlider = Scale(self.right_frame, from_=0, to=10, orient=HORIZONTAL, tickinterval=10, length=250, command=self.boxDilationIterationsChange, state=DISABLED)
        self.boxDilationIterationsSlider.set(self.boxDilationIterations)
        self.boxDilationIterationsSlider.pack()

        boxapproxLabel = Label(self.right_frame, text="Box finder sensitivity level")
        boxapproxLabel.pack()
        self.boxapproxSlider = Scale(self.right_frame, from_=0, to=0.2, orient=HORIZONTAL, tickinterval=0.2, length=250, command=self.boxapproxChange, resolution=0.01, state=DISABLED)
        self.boxapproxSlider.set(self.boxApproxParameter)
        self.boxapproxSlider.pack()

        sizeThresholdLabel = Label(self.right_frame, text="Box size exclusion threshold")
        sizeThresholdLabel.pack()
        self.sizeThresholdSlider = Scale(self.right_frame, from_=0, to=1, orient=HORIZONTAL, tickinterval=1, length=250, command=self.sizeThresholdChange, resolution=0.01, state=DISABLED)
        self.sizeThresholdSlider.set(self.sizeThreshold)
        self.sizeThresholdSlider.pack()

        duplicateThresholdLabel = Label(self.right_frame, text="Threshold for duplicate boxes")
        duplicateThresholdLabel.pack()
        self.duplicateThresholdSlider = Scale(self.right_frame, from_=0, to=100, orient=HORIZONTAL, tickinterval=100, length=250, command=self.duplicateThresholdChange, resolution=1, state=DISABLED)
        self.duplicateThresholdSlider.set(self.duplicateThreshold)
        self.duplicateThresholdSlider.pack()        

        lineDilateIterationsLabel = Label(self.right_frame, text="Line dilation iterations")
        lineDilateIterationsLabel.pack()
        self.lineDilateIterationsSlider = Scale(self.right_frame, from_=0, to=10, orient=HORIZONTAL, tickinterval=10, length=250, command=self.lineDilateIterationsChange, resolution=1, state=DISABLED)
        self.lineDilateIterationsSlider.set(self.lineDilateIterations)
        self.lineDilateIterationsSlider.pack()

        maskThicknessLabel = Label(self.right_frame, text="Thickness of box border in mask")
        maskThicknessLabel.pack()
        self.maskThicknessSlider = Scale(self.right_frame, from_=0, to=20, orient=HORIZONTAL, tickinterval=20, length=250, command=self.maskThicknessChange, resolution=1, state=DISABLED)
        self.maskThicknessSlider.set(self.maskThickness)
        self.maskThicknessSlider.pack()

        self.equalizeBackgroundCheck = Checkbutton(
            self.right_frame, text="Equalise background before processing", variable=self.equalizeBackgroundVar,
            onvalue=True, offvalue=False, command=self.equalizeBackgroundChange, state=DISABLED
        )
        if self.equalizeBackground:
            self.equalizeBackgroundCheck.select()
        self.equalizeBackgroundCheck.pack(anchor='w')

        self.skipClosingCheck = Checkbutton(
            self.right_frame, text="Skip closing", variable=self.skipClosingVar,
            onvalue=True, offvalue=False, command=self.skipClosingChange, state=DISABLED
        )
        if self.skipClosing:
            self.skipClosingCheck.select()
        self.skipClosingCheck.pack(anchor='w')

        self.unstackCheck = Checkbutton(
            self.right_frame, text="Unstack spurious links", variable=self.unstackVar,
            onvalue=True, offvalue=False, command=self.unstackChange, state=DISABLED
        )
        if self.unstack:
            self.unstackCheck.select()
        self.unstackCheck.pack(anchor='w')

        self.directionalCheck = Checkbutton(
            self.right_frame, text="Use directional heuristic", variable=self.directionalVar,
            onvalue=True, offvalue=False, command=self.directionalChange, state=DISABLED
        )
        if self.directional:
            self.directionalCheck.select()
        self.directionalCheck.pack(anchor='w')

        self.prefer_linkedCheck = Checkbutton(
            self.right_frame, text="Use prefer linked heuristic", variable=self.prefer_linkedVar,
            onvalue=True, offvalue=False, command=self.prefer_linkedChange, state=DISABLED
        )
        if self.prefer_linked:
            self.prefer_linkedCheck.select()
        self.prefer_linkedCheck.pack(anchor='w')

        self.rp_btn = Button(self.right_frame, text="Reprocess image", command=self.reprocess_image, state=DISABLED)
        self.rp_btn.pack(fill="both", expand="yes", padx="10", pady="10")

        self.generate_btn = Button(self.right_frame, text="Generate LCA model", command= self.generate_model, state=DISABLED)
        self.generate_btn.pack(fill="both", expand="yes", padx=10, pady=10)



    def select_image(self):

        self.IMAGEPATH = filedialog.askopenfilename()

        self.create_ip_object()

        self.enable_controls()

        self.reset_defaults()

        self.process_image()

    def create_ip_object(self):
        
        self.IMAGEPROCESSOR_OBJECT = ImageProcessor(self.IMAGEPATH)

    def process_image(self):    

        if len(self.IMAGEPATH) > 0:

            image = cv2.imread(self.IMAGEPATH)
            ip = self.IMAGEPROCESSOR_OBJECT
            ip.process(threshLevel=self.threshLevel, 
                       boxApproxParameter=self.boxApproxParameter, 
                       sizeThreshold=self.sizeThreshold, 
                       lineDilateIterations=self.lineDilateIterations, 
                       equalizeBackground=self.equalizeBackground, 
                       skipClosing=self.skipClosing, 
                       maskThickness=self.maskThickness,
                       skipDilation=self.skipDilation,
                       duplicateThreshold=self.duplicateThreshold,
                       boxDilationIterations=self.boxDilationIterations) 
            
            if self.unstack:
                unstack_pipeline(ip)

            if self.directional:
                directional_links_pipeline(ip)
            
            if self.prefer_linked:
                prefer_linked_pipeline(ip)

            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #edged = cv2.Canny(gray, 50, 100)
            thresh = ip.intermediates['threshold']
            dilated = ip.intermediates['closed']
            contours = ip.intermediates['contours']
            boxes = ip.intermediates['boxes']
            lines = ip.intermediates['lines']
            final = ip.intermediates['final']

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)
            #dilated = cv2.cvtColor(dilated, cv2.COLOR_BGR2RGB)
            boxes = cv2.cvtColor(boxes, cv2.COLOR_BGR2RGB)
            contours = cv2.cvtColor(contours, cv2.COLOR_BGR2RGB)
            final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)

            image = Image.fromarray(image)
            thresh = Image.fromarray(thresh)
            dilated = Image.fromarray(dilated)
            contours = Image.fromarray(contours)
            boxes = Image.fromarray(boxes)
            lines = Image.fromarray(lines)
            final = Image.fromarray(final)

            image = resize_to_square(image, self.sm_Image)
            thresh = resize_to_square(thresh, self.sm_Image)
            dilated = resize_to_square(dilated, self.sm_Image)
            contours = resize_to_square(contours, self.sm_Image)
            boxes = resize_to_square(boxes, self.sm_Image)
            lines = resize_to_square(lines, self.sm_Image)
            final = resize_to_square(final, self.lg_Image)

            image = ImageTk.PhotoImage(image)
            thresh = ImageTk.PhotoImage(thresh)
            dilated = ImageTk.PhotoImage(dilated)
            contours = ImageTk.PhotoImage(contours)
            boxes = ImageTk.PhotoImage(boxes)
            lines = ImageTk.PhotoImage(lines)
            final = ImageTk.PhotoImage(final)
            
            self.panelA1.configure(image=image)
            self.panelA2.configure(image=thresh)
            self.panelA3.configure(image=dilated)
            self.panelA4.configure(image=contours)
            self.panelA5.configure(image=boxes)
            self.panelA6.configure(image=lines)
            
            self.panelB.configure(image=final)

            self.panelA1.image = image
            self.panelA2.image = thresh
            self.panelA3.image = dilated
            self.panelA4.image = contours
            self.panelA5.image = boxes
            self.panelA6.image = lines
            
            self.panelB.image = final

            if len(ip.links) > 0:
                self.generate_btn.config(state="normal")
            else:
                self.generate_btn.config(state="disabled")

    def threshChange(self, event):
        self.threshLevel = self.threshSlider.get()

    def boxapproxChange(self, event):
        self.boxApproxParameter = self.boxapproxSlider.get()

    def lineDilateChange(self, event):
        self.lineDilateIterations = self.dilateSlider.get()

    def sizeThresholdChange(self, event):
        self.sizeThreshold = self.sizeThresholdSlider.get()

    def lineDilateIterationsChange(self, event):
        self.lineDilateIterations = self.lineDilateIterationsSlider.get()

    def maskThicknessChange(self, event):
        self.maskThickness = self.maskThicknessSlider.get()   

    def boxDilationIterationsChange(self, event):
        self.boxDilationIterations = self.boxDilationIterationsSlider.get()

    def duplicateThresholdChange(self, event):
        self.duplicateThreshold = self.duplicateThresholdSlider.get()   

    def equalizeBackgroundChange(self):
        self.equalizeBackground = self.equalizeBackgroundVar.get()

    def skipClosingChange(self):
        self.skipClosing = self.skipClosingVar.get()

    def unstackChange(self):
        self.unstack = self.unstackVar.get()

    def directionalChange(self):
        self.directional = self.directionalVar.get()

    def prefer_linkedChange(self):
        self.prefer_linked = self.prefer_linkedVar.get()

    def enable_controls(self):
        self.threshSlider.config(state="normal")
        self.boxapproxSlider.config(state="normal")
        self.sizeThresholdSlider.config(state="normal")
        self.lineDilateIterationsSlider.config(state="normal")
        self.boxDilationIterationsSlider.config(state="normal")
        self.duplicateThresholdSlider.config(state="normal")

        self.equalizeBackgroundCheck.config(state="normal")
        self.skipClosingCheck.config(state="normal")
        self.maskThicknessSlider.config(state="normal")
        self.unstackCheck.config(state="normal")
        self.directionalCheck.config(state="normal")
        self.prefer_linkedCheck.config(state="normal")

        self.rp_btn.config(state="normal")
        self.reset_btn.config(state="normal")


    def reset_defaults(self):
        self.threshLevel = self.defaults['threshLevel']
        self.threshSlider.set(self.defaults['threshLevel'])

        self.boxApproxParameter = self.defaults['boxApproxParameter']
        self.boxapproxSlider.set(self.defaults['boxApproxParameter'])

        self.sizeThreshold = self.defaults['sizeThreshold']
        self.sizeThresholdSlider.set(self.defaults['sizeThreshold'])

        self.lineDilateIterations = self.defaults['lineDilateIterations']
        self.lineDilateIterationsSlider.set(self.defaults['lineDilateIterations'])

        self.boxDilationIterations = self.defaults['boxDilationIterations']
        self.boxDilationIterationsSlider.set(self.defaults['boxDilationIterations'])

        self.duplicateThreshold = self.defaults['duplicateThreshold']
        self.duplicateThresholdSlider.set(self.defaults['duplicateThreshold'])

        self.equalizeBackground = self.defaults['equalizeBackground']
        if self.defaults['equalizeBackground']:
            self.equalizeBackgroundCheck.select()
        else:
            self.equalizeBackgroundCheck.deselect()
        
        self.skipClosing = self.defaults['skipClosing']
        if self.defaults['skipClosing']:
            self.skipClosingCheck.select()
        else:
            self.skipClosingCheck.deselect()
        
        self.maskThickness = self.defaults['maskThickness']
        self.maskThicknessSlider.set(self.defaults['maskThickness'])

        self.unstack = self.defaults['unstack']
        if self.defaults['unstack']:
            self.unstackCheck.select()
        else:
            self.unstackCheck.deselect()

        self.directional = self.defaults['directional']
        if self.defaults['directional']:
            self.directionalCheck.select()
        else:
            self.directionalCheck.deselect()

        self.prefer_linked = self.defaults['prefer_linked']
        if self.defaults['prefer_linked']:
            self.prefer_linkedCheck.select()
        else:
            self.prefer_linkedCheck.deselect()

    def reprocess_image(self):
        self.process_image()

    def generate_model(self):
        messagebox.showinfo("Information", "Informative message")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":

    app = ImageGui()

    app.run()