/**
 * Visualization functionality for Hate Speech Detection Application
 */

document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on the visualization page
    if (!document.getElementById('visualization-page')) return;
    
    // Load word cloud
    loadWordCloud();
    
    // Load charts
    loadPredictionDistribution();
    loadFeedbackAccuracy();
    loadConfidenceDistribution();
});

/**
 * Load word cloud image from the server
 */
function loadWordCloud() {
    const wordcloudContainer = document.getElementById('wordcloud-container');
    if (!wordcloudContainer) return;
    
    // Show loading spinner
    wordcloudContainer.innerHTML = '<div class="text-center py-5"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div><p class="mt-3">Generating word cloud...</p></div>';
    
    // Fetch word cloud image
    fetch('/get_wordcloud')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Create image from base64 data
            const img = document.createElement('img');
            img.src = 'data:image/png;base64,' + data.image;
            img.alt = 'Word Cloud of Hate Speech Terms';
            img.className = 'img-fluid rounded';
            
            // Clear container and append image
            wordcloudContainer.innerHTML = '';
            wordcloudContainer.appendChild(img);
        })
        .catch(error => {
            console.error('Error loading word cloud:', error);
            wordcloudContainer.innerHTML = '<div class="alert alert-danger">Failed to load word cloud. Please try again later.</div>';
        });
}

/**
 * Load prediction distribution chart
 */
function loadPredictionDistribution() {
    const chartCanvas = document.getElementById('prediction-distribution-chart');
    if (!chartCanvas) return;
    
    // Sample data - in a real app, this would come from the server
    const data = {
        labels: ['Hate Speech', 'Non-Hate Speech'],
        datasets: [{
            label: 'Number of Predictions',
            data: [35, 65],  // Sample data
            backgroundColor: [
                'rgba(220, 53, 69, 0.7)',  // Red for hate speech
                'rgba(25, 135, 84, 0.7)'   // Green for non-hate speech
            ],
            borderColor: [
                'rgba(220, 53, 69, 1)',
                'rgba(25, 135, 84, 1)'
            ],
            borderWidth: 1
        }]
    };
    
    // Create chart
    new Chart(chartCanvas, {
        type: 'pie',
        data: data,
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Prediction Distribution'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.raw || 0;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = Math.round((value / total) * 100);
                            return `${label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Load feedback accuracy chart
 */
function loadFeedbackAccuracy() {
    const chartCanvas = document.getElementById('feedback-accuracy-chart');
    if (!chartCanvas) return;
    
    // Sample data - in a real app, this would come from the server
    const data = {
        labels: ['Correct Predictions', 'Incorrect Predictions'],
        datasets: [{
            label: 'User Feedback',
            data: [85, 15],  // Sample data
            backgroundColor: [
                'rgba(13, 110, 253, 0.7)',  // Blue for correct
                'rgba(255, 193, 7, 0.7)'    // Yellow for incorrect
            ],
            borderColor: [
                'rgba(13, 110, 253, 1)',
                'rgba(255, 193, 7, 1)'
            ],
            borderWidth: 1
        }]
    };
    
    // Create chart
    new Chart(chartCanvas, {
        type: 'doughnut',
        data: data,
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Prediction Accuracy (Based on User Feedback)'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.raw || 0;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = Math.round((value / total) * 100);
                            return `${label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Load confidence distribution chart
 */
function loadConfidenceDistribution() {
    const chartCanvas = document.getElementById('confidence-distribution-chart');
    if (!chartCanvas) return;
    
    // Sample data - in a real app, this would come from the server
    const data = {
        labels: ['90-100%', '80-90%', '70-80%', '60-70%', '50-60%'],
        datasets: [{
            label: 'Number of Predictions',
            data: [45, 30, 15, 7, 3],  // Sample data
            backgroundColor: 'rgba(13, 110, 253, 0.7)',
            borderColor: 'rgba(13, 110, 253, 1)',
            borderWidth: 1
        }]
    };
    
    // Create chart
    new Chart(chartCanvas, {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Predictions'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Confidence Range'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Model Confidence Distribution'
                }
            }
        }
    });
}