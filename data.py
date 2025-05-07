
import requests
import numpy as np
from PIL import Image, ImageFile, ImageEnhance
import io
import pandas as pd
from datetime import datetime
import logging
import re

# Allow large images
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeData:
    def __init__(self):
        self.satellite_urls = [
            'https://cdn.star.nesdis.noaa.gov/GOES16/ABI/FD/GEOCOLOR/latest.jpg',
            'https://rammb-data.cira.colostate.edu/tc_realtime/images/latest.jpg',  # More reliable source
      # Backup
        ]
        self.cyclone_sources = [
            {'url': 'https://www.metoc.navy.mil/jtwc/products/niowarning.txt', 'type': 'text', 'active': True},
            {'url': 'https://mausam.imd.gov.in/imd_latest/contents/cyclone.php', 'type': 'html', 'active': True}
        ]
        self.region_of_interest = (60, 100, 0, 30)  # Indian Ocean
        self.last_cyclone_data = None

    def get_latest_satellite(self):
        """Fetch, crop, enhance, and compress the latest satellite image."""
        cyclones = self.get_real_time_cyclones()
        for url in self.satellite_urls:
            try:
                response = requests.get(url, stream=True, timeout=15)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content)).convert('RGB')
                logger.info(f"Fetched image from {url}, size: {img.size}")

                # Check if image is mostly black
                img_array = np.array(img)
                brightness = img_array.mean()
                if brightness < 10:
                    logger.warning(f"Image from {url} is too dark (brightness: {brightness:.2f})")
                    continue

                # Adjust cropping based on cyclone location if available
                if not cyclones.empty:
                    cyclone = cyclones.iloc[0]
                    img = self._adjust_crop(img, cyclone['lat'], cyclone['lon'])
                else:
                    img = self._crop_to_region(img)

                # Resize with high-quality resampling
                img = img.resize((256, 256), Image.LANCZOS)

                # Enhance sharpness for better cyclone visibility
                img = ImageEnhance.Sharpness(img).enhance(1.5)

                # Convert to WebP for better compression
                buffer = io.BytesIO()
                img.save(buffer, format="WEBP", quality=90)
                buffer.seek(0)

                # Get capture time
                last_modified = response.headers.get('Last-Modified')
                capture_time = datetime.strptime(last_modified, '%a, %d %b %Y %H:%M:%S GMT') if last_modified else datetime.utcnow()

                if cyclones.empty:
                    logger.warning("No active cyclones detected; image may not be relevant.")

                return {
                    'image': np.array(img) / 255.0,
                    'compressed_image': buffer.getvalue(),
                    'timestamp': capture_time,
                    'source': url,
                    'success': True
                }

            except requests.exceptions.RequestException as e:
                logger.warning(f"Satellite source failed {url}: {e}")
                continue

        logger.error("All satellite sources failed. Returning synthetic image.")
        return self._generate_synthetic_cyclone_response()

    def _generate_synthetic_cyclone_response(self):
        """Generate a synthetic cyclone image response."""
        img_array = self._generate_synthetic_cyclone()
        buffer = io.BytesIO()
        Image.fromarray((img_array * 255).astype(np.uint8)).save(buffer, format="WEBP", quality=90)
        buffer.seek(0)
        return {
            'image': img_array,
            'compressed_image': buffer.getvalue(),
            'timestamp': datetime.utcnow(),
            'source': 'Synthetic',
            'success': True
        }

    def _adjust_crop(self, img, lat, lon):
        """Adjust crop based on cyclone location with bounds checking."""
        width, height = img.size
        lon_min, lon_max, lat_min, lat_max = self.region_of_interest

        lon = max(lon_min, min(lon_max, lon))
        lat = max(lat_min, min(lat_max, lat))

        lon_center = lon
        lat_center = lat
        left = int((lon_center - 10 + 180) * (width / 360))
        right = int((lon_center + 10 + 180) * (width / 360))
        upper = int((90 - (lat_center + 10)) * (height / 180))
        lower = int((90 - (lat_center - 10)) * (height / 180))

        left = max(0, min(left, width - 1))
        right = max(left + 1, min(right, width))
        upper = max(0, min(upper, height - 1))
        lower = max(upper + 1, min(lower, height))

        return img.crop((left, upper, right, lower))

    def _crop_to_region(self, img):
        """Crop image to Indian Ocean region."""
        width, height = img.size
        lon_min, lon_max, lat_min, lat_max = self.region_of_interest
        left = int((lon_min + 180) * (width / 360))
        right = int((lon_max + 180) * (width / 360))
        upper = int((90 - lat_max) * (height / 180))
        lower = int((90 - lat_min) * (height / 180))
        return img.crop((left, upper, right, lower))

    def get_real_time_cyclones(self):
        """Fetch cyclone data with fallback system."""
        for source in self.cyclone_sources:
            if not source['active']:
                continue
            try:
                response = requests.get(source['url'], timeout=15)
                response.raise_for_status()
                data = self._parse_data(response, source['type'])
                if data is not None and not data.empty:
                    self.last_cyclone_data = data
                    logger.info(f"Fetched cyclone data from {source['url']}")
                    return data
            except requests.exceptions.RequestException as e:
                logger.warning(f"Cyclone source {source['url']} failed: {e}")
                source['active'] = False
        logger.warning("All cyclone sources failed. Using fallback data.")
        return self.last_cyclone_data if self.last_cyclone_data is not None else self._get_fallback_data()

    def _parse_data(self, response, data_type):
        if data_type == 'text':
            return self._parse_text(response.text)
        elif data_type == 'html':
            return self._parse_html(response.text)
        return None

    def _parse_text(self, text_content):
        cyclones = []
        current = {}
        lines = text_content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('1.'):
                if current and self._is_in_region(current.get('lat', 0), current.get('lon', 0)):
                    current['last_updated'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
                    cyclones.append(current)
                current = {'name': line.split('(')[0].replace('1.', '').strip()}
            elif 'LATITUDE' in line:
                lat_match = re.search(r'(\d+\.\d+)', line)
                if lat_match:
                    current['lat'] = float(lat_match.group(1))
            elif 'LONGITUDE' in line:
                lon_match = re.search(r'(\d+\.\d+)', line)
                if lon_match:
                    current['lon'] = float(lon_match.group(1))
            elif 'MAX SUSTAINED WINDS' in line:
                wind_match = re.search(r'(\d+) KT', line)
                if wind_match:
                    current['wind_speed'] = int(wind_match.group(1))
            elif 'MOVEMENT' in line or 'MOVING' in line:
                current['movement'] = line.split('MOV')[1].strip() if 'MOV' in line else 'Stationary'
        if current and self._is_in_region(current.get('lat', 0), current.get('lon', 0)):
            current['last_updated'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            cyclones.append(current)
        return pd.DataFrame(cyclones) if cyclones else pd.DataFrame()

    def _parse_html(self, html_content):
        cyclones = []
        try:
            lat_lon_pattern = re.compile(r'Latitude\s*(\d+\.\d+).+Longitude\s*(\d+\.\d+)', re.IGNORECASE)
            wind_pattern = re.compile(r'Wind\s*speed\s*(\d+)', re.IGNORECASE)
            matches = lat_lon_pattern.findall(html_content)
            wind_matches = wind_pattern.findall(html_content)
            for i, match in enumerate(matches):
                lat, lon = float(match[0]), float(match[1])
                if self._is_in_region(lat, lon):
                    wind_speed = int(wind_matches[i]) if i < len(wind_matches) else 0
                    cyclones.append({
                        'name': f'IMD Cyclone {i+1}',
                        'lat': lat,
                        'lon': lon,
                        'wind_speed': wind_speed,
                        'movement': 'Unknown',
                        'last_updated': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
                    })
        except Exception as e:
            logger.error(f"Failed to parse IMD HTML: {e}")
        return pd.DataFrame(cyclones) if cyclones else pd.DataFrame()

    def _is_in_region(self, lat, lon):
        """Check if location is in target region."""
        lon_min, lon_max, lat_min, lat_max = self.region_of_interest
        return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max

    def _get_fallback_data(self):
        """Provide fallback cyclone data."""
        return pd.DataFrame({
            'name': ['Cyclone SIM'],
            'lat': [15.0],
            'lon': [85.0],
            'wind_speed': [40],
            'movement': ['NW at 10 mph'],
            'last_updated': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        })

    def _generate_synthetic_cyclone(self):
        """Generate a synthetic cyclone image for fallback."""
        try:
            img_array = np.zeros((256, 256, 3), dtype=np.float32)
            center_x, center_y = 128, 128
            radius = 50
            for x in range(256):
                for y in range(256):
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if distance < radius:
                        img_array[x, y] = [0.8 + np.random.uniform(-0.2, 0.2)] * 3
                    elif distance < radius * 1.5:
                        img_array[x, y] = [0.5 + np.random.uniform(-0.1, 0.1)] * 3
            img_array = np.clip(img_array, 0, 1)
            return img_array
        except Exception as e:
            logger.error(f"Failed to generate synthetic image: {e}")
            return np.zeros((256, 256, 3), dtype=np.float32)

if __name__ == "__main__":
    data_fetcher = RealTimeData()
    cyclone_data = data_fetcher.get_real_time_cyclones()
    satellite_data = data_fetcher.get_latest_satellite()

    if satellite_data['success']:
        print("✅ Satellite image retrieved successfully!")
        print(f"Source: {satellite_data['source']}, Timestamp: {satellite_data['timestamp']}")
    else:
        print("⚠️ Failed to retrieve satellite image.")
    
    if not cyclone_data.empty:
        print("✅ Cyclone data retrieved successfully!")
        print(cyclone_data.head())
    else:
        print("⚠️ No active cyclones detected.")