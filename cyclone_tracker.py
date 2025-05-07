# cyclone_tracker.py
import requests
import json
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
from datetime import datetime
from model import TC_Model

class RealTimeCycloneTracker:
    def __init__(self):
        self.model = TC_Model(weights_path='ri_model_weights.weights.h5')
        self.nhc_url = "https://www.nhc.noaa.gov/CurrentStorms.json"
        self.satellite_base_url = "https://cdn.star.nesdis.noaa.gov/GOES19/ABI/FD/GEOCOLOR/latest.jpg"
        # https://cdn.star.nesdis.noaa.gov/GOES16/ABI/SECTOR/taw/13/
        # Atlantic Basin: Including the Gulf of Mexico, Caribbean Sea, and the North Atlantic Ocean.

        # Eastern Pacific Basin: Extending from the west coast of North America to 140°W longitude.
    
    def get_active_cyclones(self):
        """Fetch active cyclones from NHC and enhance with satellite data"""
        try:
            response = requests.get(self.nhc_url)
            response.raise_for_status()
            data = response.json()
            
            active_storms = data.get("activeStorms", [])
            enhanced_storms = []
            
            for storm in active_storms:
                # Get basic storm info
                storm_data = {
                    'name': storm.get("name", "Unnamed"),
                    'basin': storm.get("basin", "Unknown"),
                    'lat': float(storm.get("lat", 0)),
                    'lon': float(storm.get("lon", 0)),
                    'wind_speed': int(storm.get("windSpeed", 0)),
                    'pressure': int(storm.get("pressure", 1013)),
                    'movement': storm.get("movement", "Stationary"),
                    'type': storm.get("type", "Tropical Cyclone")
                }
                
                # Get satellite image for the storm
                img_data = self._get_satellite_image(storm_data['lat'], storm_data['lon'])
                storm_data['image'] = img_data['image']
                storm_data['image_url'] = img_data['image_url']
                storm_data['image_array'] = img_data['image_array']
                
                # Predict RI probability
                ri_prob = self.model.predict_with_location(
                    img_data['image_array'], 
                    storm_data['lat'], 
                    storm_data['lon']
                )
                storm_data['ri_probability'] = round(ri_prob * 100, 2)
                storm_data['risk_level'] = self._get_risk_level(ri_prob)
                
                enhanced_storms.append(storm_data)
            
            return {
                'success': True,
                'timestamp': datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                'cyclones': enhanced_storms
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            }
    
    def _get_satellite_image(self, lat, lon):
        """Get satellite image for given coordinates with storm marked"""
        try:
            # Get latest satellite image
            timestamp = datetime.utcnow().strftime("%Y%j%H%M")
            image_url = f"{self.satellite_base_url}GOES16-TAW-13-{timestamp}.jpg"
            
            response = requests.get(image_url)
            response.raise_for_status()
            
            # Process image
            img = Image.open(io.BytesIO(response.content))
            img = img.resize((512, 512))
            
            # Mark storm location on image
            draw = ImageDraw.Draw(img)
            # Convert lat/lon to image coordinates
            x = int((lon + 180) * (img.width / 360))
            y = int((90 - lat) * (img.height / 180))
            draw.ellipse([(x-10, y-10), (x+10, y+10)], outline='red', width=3)
            
            # Convert to base64 for web display
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Also return array for model prediction
            img_array = np.array(img.resize((256, 256))) / 255.0
            
            return {
                'image': f"data:image/jpeg;base64,{img_base64}",
                'image_url': image_url,
                'image_array': img_array
            }
            
        except Exception as e:
            # Fallback to synthetic image if real image fails
            synthetic_img = self._generate_synthetic_cyclone(lat, lon)
            buffered = io.BytesIO()
            synthetic_img['image'].save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            return {
                'image': f"data:image/jpeg;base64,{img_base64}",
                'image_url': "Synthetic Image",
                'image_array': synthetic_img['image_array']
            }
    
    def _generate_synthetic_cyclone(self, lat, lon):
        """Generate a synthetic cyclone image for fallback"""
        img = Image.new('RGB', (512, 512), color='darkblue')
        draw = ImageDraw.Draw(img)
        
        # Draw storm center
        x, y = 256, 256
        draw.ellipse([(x-5, y-5), (x+5, y+5)], fill='red')
        
        # Draw spiral bands
        for i in range(1, 6):
            draw.arc([x-20*i, y-20*i, x+20*i, y+20*i], 
                     start=0, end=360, fill='white', width=2)
        
        img_array = np.array(img.resize((256, 256))) / 255.0
            
        return {
            'image': img,
            'image_array': img_array
        }
    
    def _get_risk_level(self, probability):
        """Convert probability to risk level"""
        if probability >= 0.7:
            return "High"
        elif probability >= 0.4:
            return "Medium"
        else:
            return "Low"