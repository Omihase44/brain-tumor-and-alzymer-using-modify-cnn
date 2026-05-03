import cv2
import numpy as np

class ImageValidator:
    """Validates if an uploaded image is likely a medical MRI scan."""

    @staticmethod
    def is_valid_mri(image_bytes: bytes) -> bool:
        """
        Check if the image is a valid brain MRI.
        Uses heuristics like color saturation and background darkness.
        """
        try:
            # Decode the image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return False

            # Check 1: Grayscale / Low Color Saturation
            # MRIs are grayscale, so the saturation channel in HSV should be very low.
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            mean_saturation = np.mean(saturation)
            
            # If mean saturation is high, it's a colored image (e.g. regular photo)
            if mean_saturation > 30.0:
                return False

            # Check 2: Background Darkness
            # MRIs usually have black backgrounds around the edges
            height, width = image.shape[:2]
            
            # Check the four corners (10x10 patches)
            corners = [
                image[0:10, 0:10],
                image[0:10, width-10:width],
                image[height-10:height, 0:10],
                image[height-10:height, width-10:width]
            ]
            
            corner_means = [np.mean(corner) for corner in corners]
            # At least 3 corners should be very dark (typical for brain MRI)
            dark_corners = sum(1 for m in corner_means if m < 50)
            
            if dark_corners < 2:
                # Some cropped MRIs might not have dark corners, but most do.
                # Let's add a fallback: check if the center is brighter than the edges
                center = image[int(height*0.3):int(height*0.7), int(width*0.3):int(width*0.7)]
                edge_mask = np.ones((height, width), dtype=bool)
                edge_mask[int(height*0.1):int(height*0.9), int(width*0.1):int(width*0.9)] = False
                
                center_mean = np.mean(center)
                edges_mean = np.mean(image[edge_mask])
                
                if center_mean <= edges_mean + 10:
                    return False
            
            # Check 3: Aspect Ratio and Size
            # Extremely distorted or long images are likely not MRIs
            aspect_ratio = width / float(height)
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                return False
                
            return True

        except Exception as e:
            # If any processing fails, reject the image
            print(f"Image validation failed: {e}")
            return False
