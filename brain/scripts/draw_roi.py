#!/usr/bin/env python3
"""Interactive tool to draw custom ROI (Region of Interest) masks for YOLO.

Displays the first frame of a video so the user can click to draw a polygon
around the specific lanes they want YOLO to monitor (ignoring outgoing lanes).
Press 's' to save the mask as a black/white image and exit.
Press 'c' to clear the current polygon.

Usage:
    python scripts/draw_roi.py \
        --video ../datasets/ai_city_challenge/train/S01/c001/vdo.avi \
        --out-roi my_new_mask.jpg
"""

import argparse
import cv2
import numpy as np

# Global list to store polygon points
pts = []

def parse_args():
    parser = argparse.ArgumentParser(description="Draw ROI Mask Tool")
    parser.add_argument("--video", type=str, required=True, help="Input video file")
    parser.add_argument("--out-roi", type=str, default="custom_roi.jpg", help="Output mask image")
    return parser.parse_args()

def draw_polygon(event, x, y, flags, param):
    global pts
    img_display = param['img_display']
    img_clean = param['img_clean']

    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
    
    # Redraw everything
    img_display[:] = img_clean.copy()
    
    if len(pts) > 0:
        # Draw points
        for pt in pts:
            cv2.circle(img_display, pt, 5, (0, 0, 255), -1)
        # Draw lines connecting points
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                cv2.line(img_display, pts[i], pts[i+1], (0, 255, 0), 2)
            # Draw line closing the polygon back to start
            cv2.line(img_display, pts[-1], pts[0], (0, 255, 0), 2)

def main():
    global pts
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Could not open {args.video}")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Could not read first frame.")
        return

    # Create window and set mouse callback
    window_name = "Draw ROI - Click to add points, 's' to Save, 'c' to Clear, 'q' to Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # We need a clean copy to redraw over
    img_clean = frame.copy()
    img_display = frame.copy()
    
    cv2.setMouseCallback(window_name, draw_polygon, param={'img_display': img_display, 'img_clean': img_clean})

    print("=" * 60)
    print(" INSTRUCCIONES:")
    print(" 1. Hacé click izquierdo para dibujar los vértices del polígono.")
    print(" 2. Rodeá SOLO los carriles donde los autos ENTRAN a la intersección.")
    print(" 3. Presioná 'c' para borrar si te equivocaste.")
    print(" 4. Presioná 's' para GUARDAR la máscara y salir.")
    print(" 5. Presioná 'q' para salir sin guardar.")
    print("=" * 60)

    while True:
        cv2.imshow(window_name, img_display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Cancelado.")
            break
        elif key == ord('c'):
            pts.clear()
            img_display[:] = img_clean.copy()
            print("Polígono borrado.")
        elif key == ord('s'):
            if len(pts) < 3:
                print("Necesitás al menos 3 puntos para un polígono.")
                continue
            
            # Create a black mask of the same size
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            # Fill the polygon with white (255)
            poly_pts = np.array([pts], dtype=np.int32)
            cv2.fillPoly(mask, poly_pts, 255)
            
            # Save the mask
            cv2.imwrite(args.out_roi, mask)
            print(f"✅ Máscara guardada con éxito en: {args.out_roi}")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
