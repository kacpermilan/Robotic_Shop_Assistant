from recognition_module import RecognitionModule

recognition_module = RecognitionModule(tolerance=0.575)

# recognition_module.load_known_faces("known_faces")
# recognition_module.test_on_unknown_faces("unknown_faces")

recognition_module.load_known_barcodes("barcodes.csv")
recognition_module.test_on_barcode_images("barcodes")
