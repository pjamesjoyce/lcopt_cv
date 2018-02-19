from lcopt_cv.gui import ImageGui  # , DEFAULT_CONTROLS


def main():
    
    """ to add controls to the gui, import the default controls and use:
        
        controls = list(DEFAULT_CONTROLS)
        controls.append({'name': 'useless', 'type': 'checkbox', 'label': 'A useless checkbox', 'data': {'value': False}})
    
        app = ImageGui(controls=controls)
    """

    app = ImageGui()

    app.run()
