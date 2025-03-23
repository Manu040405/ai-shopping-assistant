import os
import cv2
import numpy as np

categories = ['dress', 'shoes', 'watch', 'laptop', 'shirt', 'jeans', 'kurti', 'onepiece', 'tshirt']

for category in categories:
    # Create directory if it doesn't exist
    os.makedirs(f'data/{category}', exist_ok=True)
    
    # Create 3 images for each category
    for i in range(1, 4):
        # Create a blank white image
        img = np.ones((200, 300, 3), dtype=np.uint8) * 245  # Light gray background
        
        # Add text
        text = f"{category} {i}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (100, 100, 100)  # Dark gray text
        thickness = 2
        
        # Get text size to center it
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (img.shape[0] + text_size[1]) // 2
        
        # Put text on image
        cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
        
        # Draw border
        border_color = (200, 200, 200)  # Light gray border
        cv2.rectangle(img, (0, 0), (299, 199), border_color, 2)
        
        # Save image
        cv2.imwrite(f'data/{category}/{category}_{i}.jpg', img)
        
        print(f"Created image for {category}_{i}")