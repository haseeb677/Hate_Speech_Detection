/**
 * Prediction functionality for Hate Speech Detection Application
 */

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const predictionForm = document.getElementById('detection-form');
    const textInput = document.getElementById('text-input');
    const submitButton = document.getElementById('analyze-btn');
    const resultContainer = document.getElementById('result-container');
    const resultIcon = document.getElementById('result-icon');
    const resultClass = document.getElementById('result-classification');
    const resultConfidence = document.getElementById('result-confidence');
    const resultPreprocessed = document.getElementById('result-preprocessed');
    const resultHeader = document.getElementById('result-header');
    const resultTitle = document.getElementById('result-title');
    const correctFeedbackBtn = document.getElementById('feedback-correct');
    const incorrectFeedbackBtn = document.getElementById('feedback-incorrect');
    
    // Only proceed if we're on the prediction page
    if (!predictionForm) return;
    
    // Handle form submission
    predictionForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Get the text input value
        const text = textInput.value.trim();
        
        // Validate input
        if (text.length === 0) {
            showAlert('Please enter some text to analyze.', 'warning');
            return;
        }
        
        // Disable submit button and show loading spinner
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
        
        // Hide previous results if any
        resultContainer.classList.add('d-none');
        document.getElementById('feedback-thanks').classList.add('d-none');
        
        // Make API request to the backend
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Display the results
            displayResults(data);
            
            // Re-enable submit button
            submitButton.disabled = false;
            submitButton.innerHTML = 'Analyze Text';
        })
        .catch(error => {
            console.error('Error:', error);
            showAlert('An error occurred while analyzing the text. Please try again.', 'danger');
            
            // Re-enable submit button
            submitButton.disabled = false;
            submitButton.innerHTML = 'Analyze Text';
        });
    });
    
    // Function to display prediction results
    function displayResults(data) {
        // Show result container
        resultContainer.classList.remove('d-none');
        
        // Set icon and class based on prediction
        if (data.prediction === 'Hate Speech') {
            resultIcon.innerHTML = '<i class="fas fa-exclamation-triangle text-danger"></i>';
            resultClass.textContent = 'Hate Speech';
            resultHeader.classList.remove('bg-success');
            resultHeader.classList.add('bg-danger', 'text-white');
            resultTitle.textContent = 'Hate Speech Detected';
        } else {
            resultIcon.innerHTML = '<i class="fas fa-check-circle text-success"></i>';
            resultClass.textContent = 'Non-Hate Speech';
            resultHeader.classList.remove('bg-danger');
            resultHeader.classList.add('bg-success', 'text-white');
            resultTitle.textContent = 'No Hate Speech Detected';
        }
        
        // Set confidence percentage
        if (data.probability !== null) {
            resultConfidence.textContent = (data.probability * 100).toFixed(2) + '%';
        } else {
            resultConfidence.textContent = 'N/A';
        }
        
        // Set preprocessed text if available
        if (data.preprocessed_text) {
            resultPreprocessed.textContent = data.preprocessed_text;
        }
        
        // Reset feedback thanks message
        document.getElementById('feedback-thanks').classList.add('d-none');
        
        // Enable feedback buttons
        correctFeedbackBtn.disabled = false;
        incorrectFeedbackBtn.disabled = false;
        
        // Scroll to the result
        window.scrollTo({
            top: resultContainer.offsetTop - 100,
            behavior: 'smooth'
        });
    }
    
    // Handle feedback submission
    if (correctFeedbackBtn && incorrectFeedbackBtn) {
        // Correct feedback
        correctFeedbackBtn.addEventListener('click', function() {
            submitFeedback(true);
        });
        
        // Incorrect feedback
        incorrectFeedbackBtn.addEventListener('click', function() {
            submitFeedback(false);
        });
    }
    
    // Function to submit feedback
    function submitFeedback(isCorrect) {
        // Disable both buttons
        correctFeedbackBtn.disabled = true;
        incorrectFeedbackBtn.disabled = true;
        
        // Add loading spinner to clicked button
        const clickedBtn = isCorrect ? correctFeedbackBtn : incorrectFeedbackBtn;
        const originalBtnText = clickedBtn.innerHTML;
        clickedBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Submitting...';
        
        // Get the text and prediction
        const text = textInput.value.trim();
        const prediction = resultClass.textContent;
        
        // Make API request to submit feedback
        fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                prediction: prediction,
                correct: isCorrect
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Show thank you message
            document.getElementById('feedback-thanks').classList.remove('d-none');
            
            // Disable feedback buttons
            correctFeedbackBtn.disabled = true;
            incorrectFeedbackBtn.disabled = true;
        })
        .catch(error => {
            console.error('Error:', error);
            showAlert('An error occurred while submitting feedback. Please try again.', 'danger');
            
            // Re-enable buttons
            correctFeedbackBtn.disabled = false;
            incorrectFeedbackBtn.disabled = false;
            clickedBtn.innerHTML = originalBtnText;
        });
    }
    
    // Function to show alerts
    function showAlert(message, type) {
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.role = 'alert';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Find alert container
        const alertContainer = document.getElementById('alert-container');
        if (alertContainer) {
            // Add alert to container
            alertContainer.appendChild(alertDiv);
            
            // Auto-dismiss after 5 seconds
            setTimeout(() => {
                alertDiv.classList.remove('show');
                setTimeout(() => {
                    alertDiv.remove();
                }, 150);
            }, 5000);
        }
    }
});