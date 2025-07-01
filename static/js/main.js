/**
 * Main JavaScript file for Hate Speech Detection Application
 */

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    // Add smooth scrolling for all links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 70,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Add animation to cards when they come into view
    const animateOnScroll = function() {
        const cards = document.querySelectorAll('.card');
        
        cards.forEach(card => {
            const cardPosition = card.getBoundingClientRect();
            
            // Check if card is in viewport
            if (cardPosition.top < window.innerHeight && cardPosition.bottom >= 0) {
                card.classList.add('animate__animated', 'animate__fadeInUp');
            }
        });
    };
    
    // Initial check for elements in view
    animateOnScroll();
    
    // Listen for scroll events
    window.addEventListener('scroll', animateOnScroll);
    
    // Add character counter for textarea if it exists
    const textInput = document.getElementById('text-input');
    if (textInput) {
        const createCharCounter = function() {
            // Create counter element if it doesn't exist
            if (!document.getElementById('char-counter')) {
                const counterDiv = document.createElement('div');
                counterDiv.id = 'char-counter';
                counterDiv.className = 'text-muted small text-end mt-1';
                textInput.parentNode.appendChild(counterDiv);
            }
            
            // Update counter
            const updateCounter = function() {
                const charCount = textInput.value.length;
                const counterDiv = document.getElementById('char-counter');
                counterDiv.textContent = `${charCount} characters`;
                
                // Add warning class if too long
                if (charCount > 500) {
                    counterDiv.classList.add('text-warning');
                } else {
                    counterDiv.classList.remove('text-warning');
                }
            };
            
            // Initial update
            updateCounter();
            
            // Listen for input events
            textInput.addEventListener('input', updateCounter);
        };
        
        createCharCounter();
    }
    
    // Add copy to clipboard functionality for code snippets
    document.querySelectorAll('pre code').forEach(block => {
        // Create copy button
        const button = document.createElement('button');
        button.className = 'btn btn-sm btn-outline-secondary copy-button';
        button.type = 'button';
        button.innerHTML = '<i class="fas fa-copy"></i>';
        button.title = 'Copy to clipboard';
        
        // Position the button
        const pre = block.parentNode;
        pre.style.position = 'relative';
        button.style.position = 'absolute';
        button.style.top = '5px';
        button.style.right = '5px';
        
        // Add click event
        button.addEventListener('click', function() {
            const textToCopy = block.textContent;
            navigator.clipboard.writeText(textToCopy).then(() => {
                // Change button text temporarily
                button.innerHTML = '<i class="fas fa-check"></i>';
                setTimeout(() => {
                    button.innerHTML = '<i class="fas fa-copy"></i>';
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy text: ', err);
                button.innerHTML = '<i class="fas fa-times"></i>';
                setTimeout(() => {
                    button.innerHTML = '<i class="fas fa-copy"></i>';
                }, 2000);
            });
        });
        
        pre.appendChild(button);
    });
    
    // Add dark mode toggle if it exists
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    if (darkModeToggle) {
        // Check for saved theme preference or use preferred color scheme
        const savedTheme = localStorage.getItem('theme');
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
            document.body.classList.add('dark-mode');
            darkModeToggle.checked = true;
        }
        
        // Toggle dark mode
        darkModeToggle.addEventListener('change', function() {
            if (this.checked) {
                document.body.classList.add('dark-mode');
                localStorage.setItem('theme', 'dark');
            } else {
                document.body.classList.remove('dark-mode');
                localStorage.setItem('theme', 'light');
            }
        });
    }
});