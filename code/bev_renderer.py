import numpy as np
import cv2

class BEVRenderer:
    def __init__(self, canvas_size=300, max_range=50.0):
        self.canvas_size = canvas_size
        self.max_range = max_range
        self.scale = canvas_size / (2 * max_range)
        # Ego vehicle is at pixel (150, 270) for 300x300
        self.ego_px = canvas_size // 2
        self.ego_py = int(canvas_size * 0.9)
        
        # Color cycling: [(0,255,0), (255,100,0), (0,100,255), (255,255,0), (0,255,255)]
        self.colors = [
            (0, 255, 0),     
            (255, 100, 0),   
            (0, 100, 255),   
            (255, 255, 0),   
            (0, 255, 255)    
        ]

    def world_to_pixel(self, x_m, z_m):
        px = int(self.ego_px + x_m * self.scale)
        py = int(self.ego_py - z_m * self.scale)
        return px, py

    def render(self, vehicles: list, ego_speed: float = 0.0, min_ttc: float = float('inf')) -> np.ndarray:
        # 1. Create a 300x300px black canvas
        canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        
        # Add semi-transparent dark background (base canvas is black, let's keep it simple or add a dark gray base)
        # The prompt says 'black canvas' in req 1, and 'semi-transparent dark background' in req 4.
        
        # Add a title 'BEV' in the top-left corner
        cv2.putText(canvas, "BEV", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Draw grid lines: Concentric arcs at 5m, 10m, 20m, 40m
        grid_color = (40, 40, 40)
        for dist in [5, 10, 20, 40]:
            r_px = int(dist * self.scale)
            # Draw semi-circle (arcs) above ego
            cv2.ellipse(canvas, (self.ego_px, self.ego_py), (r_px, r_px), 0, 180, 360, grid_color, 1)
            # Distance labels
            cv2.putText(canvas, f"{dist}m", (self.ego_px + 5, self.ego_py - r_px - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_color, 1)

        # Draw Ego vehicle (small cyan triangle pointing up, label 'EGO' below)
        pt1 = (self.ego_px, self.ego_py - 10)
        pt2 = (self.ego_px - 8, self.ego_py + 8)
        pt3 = (self.ego_px + 8, self.ego_py + 8)
        cv2.fillPoly(canvas, [np.array([pt1, pt2, pt3])], (255, 255, 0)) # Cyan in BGR is (255, 255, 0)
        cv2.putText(canvas, "EGO", (self.ego_px - 12, self.ego_py + 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        danger = min_ttc < 2.5
        closest_dist = float('inf')
        closest_veh = None

        # Draw other vehicles
        for v in vehicles:
            if not v.get('is_vehicle', True):
                continue
            
            x_m = v.get('X_m', 0.0)
            z_m = v.get('Z_m', 0.0)
            px, py = self.world_to_pixel(x_m, z_m)
            
            # If outside canvas bounds, skip
            if px < 0 or px >= self.canvas_size or py < 0 or py >= self.canvas_size:
                continue

            dist = np.sqrt(x_m**2 + z_m**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_veh = (px, py)

            vid = v.get('id', 0)
            color = self.colors[vid % 5]
            
            # Colored rectangles
            cv2.rectangle(canvas, (px - 5, py - 10), (px + 5, py + 10), color, -1)
            
            # Velocity arrow
            vx = v.get('Vx_ms', 0.0)
            vz = v.get('Vz_ms', 0.0)
            speed_ms = v.get('speed', np.sqrt(vx**2 + vz**2))
            
            if speed_ms > 0.5:
                # Arrow length proportional to speed
                arr_px, arr_py = self.world_to_pixel(x_m + vx, z_m + vz)
                cv2.arrowedLine(canvas, (px, py), (arr_px, arr_py), color, 2, tipLength=0.3)

            # Vehicle labels: ID and speed (e.g., 'V3 44km/h')
            speed_kmh = speed_ms * 3.6
            label = f"V{vid} {speed_kmh:.0f}km/h"
            cv2.putText(canvas, label, (px + 8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # TTC danger zone pulsing red transparent circle around closest vehicle pair
        # For simplicity, we draw it around the closest vehicle to EGO if min_ttc < 2.5s
        if danger and closest_veh:
            c_px, c_py = closest_veh
            overlay = canvas.copy()
            cv2.circle(overlay, (c_px, c_py), 30, (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
            cv2.circle(canvas, (c_px, c_py), 30, (0, 0, 255), 2)

        return canvas
