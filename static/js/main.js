document.addEventListener('DOMContentLoaded', function() {
    // Initialize soil properties section
    const autoSoilToggle = document.getElementById('auto-soil-toggle');
    const soilPropertiesManual = document.getElementById('soil-properties-manual');
    
    // Initialize variables for map
    let currentLat = null;
    let currentLng = null;
    let isManualEntry = false;
    let marker = null;
    
    // Function to toggle soil properties input fields
    function toggleSoilProperties() {
        if (autoSoilToggle && autoSoilToggle.checked) {
            if (soilPropertiesManual) {
                soilPropertiesManual.classList.add('disabled-fields');
                // If we have coordinates, fetch soil data
                if (currentLat && currentLng) {
                    getSoilData(currentLat, currentLng);
                }
            }
        } else if (soilPropertiesManual) {
            soilPropertiesManual.classList.remove('disabled-fields');
        }
    }
    
    // Add event listener to the toggle
    if (autoSoilToggle) {
        autoSoilToggle.addEventListener('change', toggleSoilProperties);
        // Initialize on page load
        toggleSoilProperties();
    }
    
    // Function to get soil data based on location
    function getSoilData(lat, lon) {
        // Check if soil properties section exists
        if (!soilPropertiesManual) {
            console.error('Soil properties section not found');
            return;
        }
        
        // Show loading spinner in soil properties section
        soilPropertiesManual.innerHTML = `
            <div class="text-center py-3">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2">Estimating soil properties based on location...</p>
            </div>
        `;
        
        // Determine soil properties based on latitude
        // This is a simplified approach that mimics what the server would do
        let soil_ph = 6.5;
        let nitrogen = 40;
        let phosphorus = 30;
        let potassium = 40;
        let organic_matter = 2.5;
        let soil_type = "Loam";
        
        // Adjust based on region/climate using latitude
        if (Math.abs(lat) < 15) {  // Tropical regions
            soil_ph = 5.8;  // More acidic in tropical regions
            nitrogen = 35;
            phosphorus = 25;
            potassium = 50;
            organic_matter = 3.2;
            soil_type = "Clay";
        } else if (Math.abs(lat) > 40) {  // Arid/temperate regions
            soil_ph = 7.2;  // More alkaline in arid regions
            nitrogen = 25;
            phosphorus = 20;
            potassium = 30;
            organic_matter = 1.8;
            soil_type = "Sandy";
        } else if (15 <= Math.abs(lat) && Math.abs(lat) < 30) {  // Subtropical regions
            soil_ph = 6.2;
            nitrogen = 45;
            phosphorus = 35;
            potassium = 45;
            organic_matter = 2.8;
            soil_type = "Loam";
        }
        
        // Create a data object that mimics the API response
        const data = {
            soil_ph: soil_ph,
            nitrogen: nitrogen,
            phosphorus: phosphorus,
            potassium: potassium,
            organic_matter: organic_matter,
            soil_type: soil_type,
            location_name: "Estimated location"
        };
        
        // Restore the original HTML structure
        soilPropertiesManual.innerHTML = `
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label for="soil_ph" class="form-label">Soil pH</label>
                    <div class="input-group">
                        <span class="input-group-text"><i class="fas fa-vial"></i></span>
                        <input type="number" step="0.1" class="form-control" id="soil_ph" name="soil_ph" required min="0" max="14" placeholder="6.5" value="${data.soil_ph}">
                    </div>
                    <small class="text-muted">Typical range: 5.5 - 7.5</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label for="nitrogen" class="form-label">Nitrogen (N)</label>
                    <div class="input-group">
                        <span class="input-group-text">N</span>
                        <input type="number" class="form-control" id="nitrogen" name="nitrogen" required min="0" max="200" placeholder="40" value="${data.nitrogen}">
                    </div>
                    <small class="text-muted">Measured in kg/ha</small>
                </div>
            </div>

            <div class="row">
                <div class="col-md-6 mb-3">
                    <label for="phosphorus" class="form-label">Phosphorus (P)</label>
                    <div class="input-group">
                        <span class="input-group-text">P</span>
                        <input type="number" class="form-control" id="phosphorus" name="phosphorus" required min="0" max="200" placeholder="30" value="${data.phosphorus}">
                    </div>
                    <small class="text-muted">Measured in kg/ha</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label for="potassium" class="form-label">Potassium (K)</label>
                    <div class="input-group">
                        <span class="input-group-text">K</span>
                        <input type="number" class="form-control" id="potassium" name="potassium" required min="0" max="600" placeholder="40" value="${data.potassium}">
                    </div>
                    <small class="text-muted">Measured in kg/ha</small>
                </div>
            </div>

            <div class="row">
                <div class="col-md-6 mb-3">
                    <label for="organic_matter" class="form-label">Organic Matter (%)</label>
                    <div class="input-group">
                        <span class="input-group-text"><i class="fas fa-leaf"></i></span>
                        <input type="number" step="0.1" class="form-control" id="organic_matter" name="organic_matter" required min="0" max="10" placeholder="2.5" value="${data.organic_matter}">
                    </div>
                    <small class="text-muted">Typical range: 1% - 5%</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label for="soil_type" class="form-label">Soil Type</label>
                    <select class="form-select" id="soil_type" name="soil_type" required>
                        <option value="" disabled>Select soil type</option>
                        <option value="Clay" ${data.soil_type === 'Clay' ? 'selected' : ''}>Clay</option>
                        <option value="Loam" ${data.soil_type === 'Loam' ? 'selected' : ''}>Loam</option>
                        <option value="Sandy" ${data.soil_type === 'Sandy' ? 'selected' : ''}>Sandy</option>
                        <option value="Silt" ${data.soil_type === 'Silt' ? 'selected' : ''}>Silt</option>
                    </select>
                </div>
            </div>
        `;
        
        // Add disabled class if auto-detect is on
        if (autoSoilToggle && autoSoilToggle.checked) {
            soilPropertiesManual.classList.add('disabled-fields');
        }
        
        // Show a success notification
        const soilDataAlert = document.createElement('div');
        soilDataAlert.className = 'alert alert-success mt-2 mb-3 fade show';
        soilDataAlert.innerHTML = `
            <i class="fas fa-check-circle me-2"></i>
            Soil properties estimated based on your location
            <button type="button" class="btn-close float-end" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        soilPropertiesManual.parentNode.insertBefore(soilDataAlert, soilPropertiesManual);
        
        // Remove the alert after 5 seconds
        setTimeout(() => {
            soilDataAlert.classList.remove('show');
            setTimeout(() => soilDataAlert.remove(), 150);
        }, 5000);
    }
    
    // Handle contact form submission
    const contactForm = document.getElementById('contact-form');
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const sendButton = document.getElementById('send-message');
            sendButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Sending...';
            sendButton.disabled = true;
            
            // Get form data
            const formData = {
                name: document.getElementById('name').value,
                email: document.getElementById('email').value,
                subject: document.getElementById('subject').value,
                message: document.getElementById('message').value
            };
            
            // Send data to server using fetch API
            fetch('/contact', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData),
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    sendButton.innerHTML = '<i class="fas fa-check-circle me-2"></i>Sent!';
                    document.getElementById('contact-success').classList.remove('d-none');
                    
                    // Reset form after a delay
                    setTimeout(() => {
                        contactForm.reset();
                        sendButton.innerHTML = '<i class="fas fa-paper-plane me-2"></i>Send Message';
                        sendButton.disabled = false;
                        
                        // Close modal after submission
                        const contactModal = bootstrap.Modal.getInstance(document.getElementById('contactModal'));
                        if (contactModal) {
                            contactModal.hide();
                        }
                        
                        // Hide success message after modal is closed
                        setTimeout(() => {
                            document.getElementById('contact-success').classList.add('d-none');
                        }, 500);
                    }, 2000);
                } else {
                    // Handle error
                    sendButton.innerHTML = '<i class="fas fa-paper-plane me-2"></i>Send Message';
                    sendButton.disabled = false;
                    alert('Error sending message: ' + data.message);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                sendButton.innerHTML = '<i class="fas fa-paper-plane me-2"></i>Send Message';
                sendButton.disabled = false;
                alert('Error sending message. Please try again later.');
            });
        });
    }
    
    // Add animation effects to cards
    const animateCards = () => {
        const cards = document.querySelectorAll('.fade-in');
        cards.forEach((card, index) => {
            setTimeout(() => {
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, 100 * index);
        });
    };
    
    // Initialize animations
    animateCards();
    
    // Add dynamic weather background based on current conditions
    const updateWeatherBackground = (weatherData) => {
        const weatherInfo = document.getElementById('weather-info');
        if (!weatherInfo) return;
        
        let bgClass = 'bg-clear';
        
        if (weatherData.rainfall > 5) {
            bgClass = 'bg-rainy';
        } else if (weatherData.temperature > 30) {
            bgClass = 'bg-hot';
        } else if (weatherData.temperature < 10) {
            bgClass = 'bg-cold';
        } else if (weatherData.humidity > 80) {
            bgClass = 'bg-humid';
        }
        
        // Remove all previous bg classes
        weatherInfo.classList.remove('bg-clear', 'bg-rainy', 'bg-hot', 'bg-cold', 'bg-humid');
        // Add new bg class
        weatherInfo.classList.add(bgClass);
    };
    // Initialize the map with a beautiful tile layer
    let map;
    const mapElement = document.getElementById('map');
    
    if (mapElement) {
        map = L.map('map').setView([20.5937, 78.9629], 5); // Default view centered on India
        
        // Add a more visually appealing tile layer
        L.tileLayer('https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, Tiles style by <a href="https://www.hotosm.org/" target="_blank">HOT</a>'
        }).addTo(map);
        
        // Add click event to map
        map.on('click', function(e) {
            const lat = e.latlng.lat;
            const lng = e.latlng.lng;
            
            // Remove existing marker if any
            if (marker) {
                map.removeLayer(marker);
            }
            
            // Add new marker
            marker = L.marker([lat, lng]).addTo(map);
            marker.bindPopup(`
                <div class="text-center">
                    <h6 class="mb-1">Selected Location</h6>
                    <p class="mb-0 small">Lat: ${lat.toFixed(4)}, Lng: ${lng.toFixed(4)}</p>
                </div>
            `).openPopup();
            
            // Get location details
            getLocationDetails(lat, lng);
            
            // Enable the predict button
            const predictButton = document.getElementById('predict-button');
            if (predictButton) {
                predictButton.disabled = false;
            }
        });
    }
    
    // Function to get location details using OpenCage
    function getLocationDetails(lat, lon) {
        // Store current coordinates
        currentLat = lat;
        currentLng = lon;
        
        // Show loading spinner in location details
        document.getElementById('location-data').innerHTML = `
            <div class="col-12 text-center py-3">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2">Fetching location details...</p>
            </div>
        `;
        document.getElementById('location-details').classList.remove('d-none');
        
        fetch('/get_location_details', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ lat, lon }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error:', data.error);
                document.getElementById('location-details').classList.add('d-none');
                return;
            }
            
            // Update form fields with location data
            const regionSelect = document.getElementById('region');
            const elevationInput = document.getElementById('elevation');
            
            // Set region type
            for (let i = 0; i < regionSelect.options.length; i++) {
                if (regionSelect.options[i].value === data.region_type) {
                    regionSelect.selectedIndex = i;
                    break;
                }
            }
            
            // Set elevation
            elevationInput.value = data.elevation;
            
            // Display location information
            const locationData = document.getElementById('location-data');
            locationData.innerHTML = `
                <div class="col-md-6 mb-2">
                    <div class="d-flex align-items-center">
                        <i class="fas fa-map-marker-alt me-2 text-primary"></i>
                        <div>
                            <small class="text-muted">Location</small>
                            <p class="mb-0 fw-medium">${data.location_name}</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-6 mb-2">
                    <div class="d-flex align-items-center">
                        <i class="fas fa-globe-americas me-2 text-primary"></i>
                        <div>
                            <small class="text-muted">Country/State</small>
                            <p class="mb-0 fw-medium">${data.country}${data.state ? ', ' + data.state : ''}</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-6 mb-2">
                    <div class="d-flex align-items-center">
                        <i class="fas fa-mountain me-2 text-primary"></i>
                        <div>
                            <small class="text-muted">Elevation</small>
                            <p class="mb-0 fw-medium">${data.elevation} meters</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-6 mb-2">
                    <div class="d-flex align-items-center">
                        <i class="fas fa-cloud-sun me-2 text-primary"></i>
                        <div>
                            <small class="text-muted">Region Type</small>
                            <p class="mb-0 fw-medium">${data.region_type}</p>
                        </div>
                    </div>
                </div>
            `;
            
            document.getElementById('location-details').classList.remove('d-none');
            
            // Now get weather data
            updateWeather(lat, lon);
            
            // If auto-detect soil properties is enabled, get soil data
            if (autoSoilToggle && autoSoilToggle.checked) {
                getSoilData(lat, lon);
            }
            
            // Enable the predict button
            const predictButton = document.getElementById('predict-button');
            if (predictButton) {
                predictButton.disabled = false;
            }
        })
        .catch(error => {
            console.error('Error fetching location data:', error);
            document.getElementById('location-details').classList.add('d-none');
            
            // Set default values for region and elevation
            const regionField = document.getElementById('region');
            const elevationField = document.getElementById('elevation');
            
            if (regionField) regionField.value = 'Temperate'; // Default region
            if (elevationField) elevationField.value = 100; // Default elevation
            
            // Set default weather values
            setDefaultWeatherValues();
            
            // Enable the predict button despite the error
            const predictButton = document.getElementById('predict-button');
            if (predictButton) {
                predictButton.disabled = false;
            }
            
            // Show an error notification
            alert('Could not fetch location details. Default values have been set.');
        });
    }
    
    // Function to update weather information
    function updateWeather(lat, lon) {
        // Check if weather elements exist
        const weatherDetails = document.getElementById('weather-details');
        const weatherInfo = document.getElementById('weather-info');
        
        if (!weatherDetails || !weatherInfo) {
            console.error('Weather elements not found in the DOM');
            return;
        }
        
        // Show loading spinner in weather info
        weatherDetails.innerHTML = `
            <div class="col-12 text-center py-3">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2">Fetching weather data...</p>
            </div>
        `;
        weatherInfo.classList.remove('d-none');
        
        fetch('/get_weather', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ lat, lon }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error:', data.error);
                weatherInfo.classList.add('d-none');
                return;
            }
            
            // Update hidden form fields with weather data
            const rainfallInput = document.getElementById('rainfall');
            const temperatureInput = document.getElementById('temperature');
            const humidityInput = document.getElementById('humidity');
            const sunlightInput = document.getElementById('sunlight');
            const weatherLocationBadge = document.getElementById('weather-location');
            
            if (rainfallInput) rainfallInput.value = data.rainfall;
            if (temperatureInput) temperatureInput.value = data.temperature;
            if (humidityInput) humidityInput.value = data.humidity;
            if (sunlightInput) sunlightInput.value = data.sunlight;
            
            // Update location badge
            if (weatherLocationBadge) weatherLocationBadge.textContent = data.location;
            weatherDetails.innerHTML = `
                <div class="col-md-6 mb-2">
                    <div class="d-flex align-items-center">
                        <i class="fas fa-thermometer-half me-2 text-danger"></i>
                        <div>
                            <small class="text-muted">Temperature</small>
                            <p class="mb-0 fw-medium">${data.temperature}°C</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-6 mb-2">
                    <div class="d-flex align-items-center">
                        <i class="fas fa-tint me-2 text-primary"></i>
                        <div>
                            <small class="text-muted">Humidity</small>
                            <p class="mb-0 fw-medium">${data.humidity}%</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-6 mb-2">
                    <div class="d-flex align-items-center">
                        <i class="fas fa-cloud-rain me-2 text-info"></i>
                        <div>
                            <small class="text-muted">Rainfall</small>
                            <p class="mb-0 fw-medium">${data.rainfall} mm</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-6 mb-2">
                    <div class="d-flex align-items-center">
                        <i class="fas fa-sun me-2 text-warning"></i>
                        <div>
                            <small class="text-muted">Sunlight</small>
                            <p class="mb-0 fw-medium">${data.sunlight.toFixed(1)} hours</p>
                        </div>
                    </div>
                </div>
            `;
            
            document.getElementById('weather-info').classList.remove('d-none');
            
            // Update weather background based on conditions
            updateWeatherBackground(data);
            
            // Enable the predict button if not in manual entry mode
            if (!isManualEntry) {
                document.getElementById('predict-button').disabled = false;
            }
            
            // Add a subtle animation to the weather card
            const weatherCard = document.getElementById('weather-info');
            weatherCard.classList.add('weather-updated');
            setTimeout(() => {
                weatherCard.classList.remove('weather-updated');
            }, 1000);
        })
        .catch(error => {
            console.error('Error fetching weather data:', error);
            if (weatherInfo) {
                weatherInfo.classList.add('d-none');
            }
            
            // Set default weather values
            setDefaultWeatherValues();
        });
    }
    
    // Handle map clicks
    map.on('click', function(e) {
        const lat = e.latlng.lat;
        const lng = e.latlng.lng;
        
        // Remove existing marker if any
        if (marker) {
            map.removeLayer(marker);
        }
        
        // Add a new marker with a custom icon
        marker = L.marker([lat, lng], {
            title: 'Selected Location',
            alt: 'Selected Location',
            riseOnHover: true
        }).addTo(map);
        
        marker.bindPopup(`
            <div class="text-center">
                <h6 class="mb-1">Selected Location</h6>
                <p class="mb-0 small">Lat: ${lat.toFixed(4)}, Lng: ${lng.toFixed(4)}</p>
            </div>
        `).openPopup();
        
        // Reset manual entry mode
        isManualEntry = false;
        document.getElementById('manual-entry-section').classList.add('d-none');
        
        // Get location details (which will also get weather)
        getLocationDetails(lat, lng);
    });
    
    // Handle location search
    document.getElementById('search-button').addEventListener('click', function() {
        const searchQuery = document.getElementById('location-search').value;
        if (!searchQuery) return;
        
        // Show loading state
        this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Searching...';
        this.disabled = true;
        
        // Use Nominatim for geocoding (free and open-source)
        fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchQuery)}&limit=1`)
            .then(response => response.json())
            .then(data => {
                // Reset button state
                document.getElementById('search-button').innerHTML = 'Search';
                document.getElementById('search-button').disabled = false;
                
                if (data && data.length > 0) {
                    const location = data[0];
                    const lat = parseFloat(location.lat);
                    const lon = parseFloat(location.lon);
                    
                    // Center map on the location
                    map.setView([lat, lon], 10);
                    
                    // Remove existing marker if any
                    if (marker) {
                        map.removeLayer(marker);
                    }
                    
                    // Add a new marker
                    marker = L.marker([lat, lon]).addTo(map);
                    marker.bindPopup(`
                        <div class="text-center">
                            <h6 class="mb-1">${location.display_name.split(',', 2).join(',')}</h6>
                            <p class="mb-0 small">Lat: ${lat.toFixed(4)}, Lng: ${lon.toFixed(4)}</p>
                        </div>
                    `).openPopup();
                    
                    // Reset manual entry mode
                    isManualEntry = false;
                    document.getElementById('manual-entry-section').classList.add('d-none');
                    
                    // Get location details
                    getLocationDetails(lat, lon);
                } else {
                    // Show error message
                    alert('Location not found. Please try a different search term.');
                }
            })
            .catch(error => {
                // Reset button state
                document.getElementById('search-button').innerHTML = 'Search';
                document.getElementById('search-button').disabled = false;
                
                console.error('Error searching for location:', error);
                alert('Error searching for location. Please try again.');
            });
    });
    
    // Allow pressing Enter in the search box
    document.getElementById('location-search').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            document.getElementById('search-button').click();
            e.preventDefault();
        }
    });
    
    // Set default values for weather data if API fails
    function setDefaultWeatherValues() {
        const rainfallInput = document.getElementById('rainfall');
        const temperatureInput = document.getElementById('temperature');
        const humidityInput = document.getElementById('humidity');
        const sunlightInput = document.getElementById('sunlight');
        
        if (rainfallInput && !rainfallInput.value) {
            rainfallInput.value = 50; // Default rainfall
        }
        if (temperatureInput && !temperatureInput.value) {
            temperatureInput.value = 25; // Default temperature
        }
        if (humidityInput && !humidityInput.value) {
            humidityInput.value = 60; // Default humidity
        }
        if (sunlightInput && !sunlightInput.value) {
            sunlightInput.value = 8; // Default sunlight hours
        }
    }
    
    // Form submission loading state
    const cropForm = document.getElementById('crop-form');
    if (cropForm) {
        cropForm.addEventListener('submit', function() {
            // Show loading modal
            const loadingModalElement = document.getElementById('loadingModal');
            if (loadingModalElement) {
                const loadingModal = new bootstrap.Modal(loadingModalElement);
                loadingModal.show();
            }
        });
    }
    
    // Try to get user's location on page load
    if (navigator.geolocation && map) {
        navigator.geolocation.getCurrentPosition(
            function(position) {
                const lat = position.coords.latitude;
                const lng = position.coords.longitude;
                
                // Set map view to user location
                map.setView([lat, lng], 10);
                
                // Add marker
                marker = L.marker([lat, lng]).addTo(map);
                marker.bindPopup(`
                    <div class="text-center">
                        <h6 class="mb-1">Your Location</h6>
                        <p class="mb-0 small">Lat: ${lat.toFixed(4)}, Lng: ${lng.toFixed(4)}</p>
                    </div>
                `).openPopup();
                
                // Get location details
                getLocationDetails(lat, lng);
                
                // Enable the predict button
                const predictButton = document.getElementById('predict-button');
                if (predictButton) {
                    predictButton.disabled = false;
                }
            },
            function(error) {
                console.log("Error getting location: ", error.message);
                // No need to show an error - the default map view will be shown
            }
        );
    }
});