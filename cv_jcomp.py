import cv2
import numpy as np
import os
import sys
import time

class TrafficLightController:
    def __init__(self, green_duration=30):
        """
        Initialize the traffic light controller
        
        Args:
            green_duration: Duration of green light in seconds
        """
        self.green_duration = green_duration
        self.current_green = None  # Index of current green light
        self.timer_start = None    # Timestamp when green light started
        self.remaining_time = 0    # Remaining time for current green light
    
    def get_traffic_lights(self, vehicle_counts):
        """
        Determine traffic light states based on vehicle counts
        
        Args:
            vehicle_counts: List of vehicle counts for each intersection
            
        Returns:
            List of traffic light states ("GREEN" or "RED") for each intersection,
            remaining time, and whether a new green light was assigned
        """
        current_time = time.time()
        
        # Initialize lights to all red
        lights = ["RED"] * len(vehicle_counts)
        new_green_assigned = False
        
        # Check if we need to start a new green cycle
        if self.current_green is None or (current_time - self.timer_start >= self.green_duration):
            # Find intersection with maximum vehicles
            max_count = max(vehicle_counts)
            max_indices = [i for i, count in enumerate(vehicle_counts) if count == max_count]
            
            # If there's a tie, choose the one that hasn't been green recently
            self.current_green = max_indices[0]
            
            # Start the timer
            self.timer_start = current_time
            self.remaining_time = self.green_duration
            new_green_assigned = True
        else:
            # Update remaining time
            self.remaining_time = self.green_duration - (current_time - self.timer_start)
        
        # Set the current green light
        lights[self.current_green] = "GREEN"
        
        return lights, int(self.remaining_time), new_green_assigned


class ImprovedVehicleDetection:
    def __init__(self, sources, skip_frames=2, green_duration=30):
        self.sources = sources
        self.skip_frames = skip_frames
        self.frame_count = 0
        
        # Calculate grid dimensions
        self.grid_cols = min(2, len(sources))  # Max 2 columns
        self.grid_rows = (len(sources) + self.grid_cols - 1) // self.grid_cols  # Ceiling division
        
        # Prepare video writer (adjust dimensions for grid layout)
        self.frame_width = int(sources[0].get(3))
        self.frame_height = int(sources[0].get(4))
        
        # Set output dimensions based on grid
        self.output_width = self.frame_width * self.grid_cols
        self.output_height = self.frame_height * self.grid_rows
        
        self.out = cv2.VideoWriter(
            "output.mp4", 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            30,  # Frame rate
            (self.output_width, self.output_height)
        )
        
        # Initialize background subtractors for each source
        self.bg_subtractors = []
        for _ in sources:
            self.bg_subtractors.append(cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=True))
        
        # Initialize vehicle trackers for each source
        self.prev_vehicles = []
        for _ in sources:
            self.prev_vehicles.append([])
        
        # Initialize vehicle counts
        self.vehicle_counts = [0] * len(sources)
        
        # Initialize traffic light controller
        self.traffic_controller = TrafficLightController(green_duration=green_duration)
        
    def detect_vehicles(self, frame, bg_subtractor, prev_vehicles):
        """
        Detect vehicles using background subtraction and contour analysis
        with temporal consistency
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(blurred)
        
        # Remove shadows (they're marked as 127 in MOG2)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to remove noise and fill gaps
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours that might be vehicles
        vehicles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip small contours
            if area < 800:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Aspect ratio and size filtering
            aspect_ratio = w / float(h)
            if (area > 800 and  # Minimum area
                area < 30000 and  # Maximum area
                0.4 < aspect_ratio < 4.0):  # Reasonable vehicle aspect ratio
                
                vehicles.append((x, y, w, h, area))
        
        # Sort by area (larger vehicles are more likely to be real)
        vehicles.sort(key=lambda v: v[4], reverse=True)
        
        # Apply temporal consistency
        final_vehicles = []
        for v in vehicles:
            x, y, w, h, area = v
            
            # Check if this vehicle overlaps with any previous frame vehicle
            is_new = True
            for pv in prev_vehicles:
                px, py, pw, ph = pv
                
                # Check for overlap
                if (x < px + pw and x + w > px and
                    y < py + ph and y + h > py):
                    is_new = False
                    # Use the more stable detection
                    final_vehicles.append((x, y, w, h))
                    break
            
            if is_new and len(final_vehicles) < 20:  # Limit max vehicles to prevent false positives
                final_vehicles.append((x, y, w, h))
        
        return final_vehicles
    
    def draw_traffic_light(self, frame, status, remaining_time):
        """
        Draw traffic light indicator on the frame
        
        Args:
            frame: The video frame to draw on
            status: "GREEN" or "RED"
            remaining_time: Remaining time in seconds for the current light
        """
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Traffic light background
        light_width = 80
        light_height = 180
        light_x = width - light_width - 20
        light_y = 20
        
        # Draw traffic light housing
        cv2.rectangle(frame, (light_x, light_y), 
                     (light_x + light_width, light_y + light_height), 
                     (50, 50, 50), -1)
        cv2.rectangle(frame, (light_x, light_y), 
                     (light_x + light_width, light_y + light_height), 
                     (0, 0, 0), 2)
        
        # Draw red light
        red_center = (light_x + light_width // 2, light_y + 40)
        red_radius = 25
        red_color = (0, 0, 255) if status == "RED" else (0, 0, 50)
        cv2.circle(frame, red_center, red_radius, red_color, -1)
        cv2.circle(frame, red_center, red_radius, (0, 0, 0), 2)
        
        # Draw green light
        green_center = (light_x + light_width // 2, light_y + light_height - 40)
        green_radius = 25
        green_color = (0, 255, 0) if status == "GREEN" else (0, 50, 0)
        cv2.circle(frame, green_center, green_radius, green_color, -1)
        cv2.circle(frame, green_center, green_radius, (0, 0, 0), 2)
        
        # Draw timer (only for green light)
        if status == "GREEN":
            cv2.putText(frame, f"{remaining_time}s", 
                        (light_x, light_y + light_height + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def process_video(self):
        self.frame_count = 0
        while True:
            frames = []
            processed_frames = []
            
            # Always process detection every nth frame, regardless of light status
            process_detection = (self.frame_count % self.skip_frames == 0)
            
            # Get current traffic light statuses
            lights, remaining_time, new_green_assigned = self.traffic_controller.get_traffic_lights(self.vehicle_counts)
            
            for i, cap in enumerate(self.sources):
                ret, frame = cap.read()
                if not ret:
                    return
                
                # More intelligent downscaling
                frame = cv2.resize(frame, (720, 300), 
                                   interpolation=cv2.INTER_AREA)
                frames.append(frame.copy())
                
                if process_detection:
                    # Detect vehicles using improved method
                    vehicle_detections = self.detect_vehicles(
                        frame, self.bg_subtractors[i], self.prev_vehicles[i])
                    
                    # Update previous vehicles for tracking
                    self.prev_vehicles[i] = [(x, y, w, h) for (x, y, w, h) in vehicle_detections]
                    
                    # Update vehicle count for this source
                    # KEY CHANGE: Always updating counts regardless of traffic light state
                    self.vehicle_counts[i] = len(vehicle_detections)
                    
                    # Only reset the count if this intersection just got a green light
                    if lights[i] == "GREEN" and new_green_assigned and i == self.traffic_controller.current_green:
                        self.vehicle_counts[i] = 0
                    
                    # Draw detections with different colors based on light status
                    for (x, y, w, h) in vehicle_detections:
                        # Green rectangle for green light, red for red light
                        rect_color = (0, 255, 0) if lights[i] == "GREEN" else (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), rect_color, 2)
                
                # Add vehicle count text
                cv2.putText(frame, f'Vehicles: {self.vehicle_counts[i]}', 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Draw traffic light indicator
                self.draw_traffic_light(frame, lights[i], remaining_time)
                
                # Draw traffic status text
                cv2.putText(frame, f'Status: {lights[i]}', 
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (0, 255, 0) if lights[i] == "GREEN" else (0, 0, 255), 2)
                
                # Draw intersection number
                cv2.putText(frame, f'Intersection #{i+1}', 
                           (20, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                processed_frames.append(frame)
            
            # Increment frame count once per loop, not per source
            self.frame_count += 1
            
            # Create a grid layout of frames
            # First, ensure we have a complete grid by adding blank frames if needed
            while len(processed_frames) < self.grid_rows * self.grid_cols:
                blank_frame = np.zeros((300, 720, 3), dtype=np.uint8)
                processed_frames.append(blank_frame)
            
            # Create rows
            rows = []
            for r in range(self.grid_rows):
                start_idx = r * self.grid_cols
                end_idx = start_idx + self.grid_cols
                row_frames = processed_frames[start_idx:end_idx]
                rows.append(np.hstack(row_frames))
            
            # Stack rows vertically
            merged_frame = np.vstack(rows)
            
            # Add overall timer text
            cv2.putText(merged_frame, f'Next light change: {remaining_time}s', 
                       (20, merged_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Display and write
            cv2.imshow("Traffic Light Control System", merged_frame)
            self.out.write(merged_frame)
            
            # Quick exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def run(self):
        try:
            self.process_video()
        finally:
            # Cleanup
            for cap in self.sources:
                cap.release()
            self.out.release()
            cv2.destroyAllWindows()


# Enhanced video source selection function
def get_video_source():
    """
    Prompt user to choose video source type
    Returns a list of video capture objects
    """
    sources = []
    num_sources = int(input("Enter the number of video sources (intersections): "))
    
    for i in range(num_sources):
        print("\nChoose source type for video source", i+1)
        print("1. Webcam")
        print("2. IP Camera")
        print("3. Video File")
        
        choice = input("Enter your choice (1/2/3): ").strip()
        
        if choice == '1':  # Webcam
            # Try multiple common webcam indices
            for idx in [3,1,0,2]:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    print(f"Successfully opened webcam at index {idx}")
                    sources.append(cap)
                    break
            else:
                print(f"Could not open webcam for source {i+1}")
        
        elif choice == '2':  # IP Camera
            url = input(f"Enter IP camera URL for video source {i+1}: ")
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                sources.append(cap)
                print(f"Successfully connected to IP camera")
            else:
                print(f"Could not open IP camera for source {i+1}")
        
        elif choice == '3':  # Video File
            while True:
                file_path = input(f"Enter full path to video file for source {i+1}: ").strip()
                file_path = file_path.strip('\'"')
                if os.path.exists(file_path):
                    cap = cv2.VideoCapture(file_path)
                    if cap.isOpened():
                        sources.append(cap)
                        print(f"Successfully opened video file: {file_path}")
                        break
                    else:
                        print("Could not open video file. Please check the file format.")
                else:
                    print("File does not exist. Please enter a valid file path.")
        
        else:
            print("Invalid choice. Please try again.")
            i -= 1  # Retry this source
    
    return sources


def main():
    print("Traffic Light Control System")
    print("===========================")
    
    # Get video sources
    sources = get_video_source()
    
    # Check if any sources were successfully opened
    if not sources:
        print("No video sources could be opened. Exiting.")
        return
    
    print(f"Successfully opened {len(sources)} video source(s)")
    
    # Get green light duration
    try:
        green_duration = int(input("Enter green light duration in seconds (default: 30): ") or "30")
    except ValueError:
        print("Invalid input. Using default duration of 30 seconds.")
        green_duration = 30
    
    # Initialize processor with traffic light control
    processor = ImprovedVehicleDetection(sources, skip_frames=2, green_duration=green_duration)
    print("Press 'q' to quit the application")
    processor.run()


if __name__ == "__main__":
    main()
