// JavaScript for Nitel Araştırma Örneklem Tahmin Sistemi

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const predictionResult = document.getElementById('prediction-result');
    
    // Form submission handler
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading state
        showLoading();
        
        try {
            // Get form data
            const formData = new FormData(form);
            
            // Send prediction request
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                showPrediction(result.predicted_sample_size, result.confidence);
            } else {
                showError(result.error);
            }
            
        } catch (error) {
            showError('Bir hata oluştu. Lütfen tekrar deneyin.');
            console.error('Error:', error);
        }
    });
    
    // Show loading state
    function showLoading() {
        predictionResult.innerHTML = `
            <div class="text-center">
                <div class="loading mb-3"></div>
                <h4 class="text-primary">Tahmin Yapılıyor...</h4>
                <p class="text-muted">Lütfen bekleyin</p>
            </div>
        `;
    }
    
    // Show prediction result
    function showPrediction(sampleSize, confidence) {
        const confidenceClass = getConfidenceClass(confidence);
        
        predictionResult.innerHTML = `
            <div class="prediction-result">
                <div class="prediction-number">${sampleSize}</div>
                <div class="prediction-label">Örneklem Büyüklüğü</div>
                <div class="prediction-confidence ${confidenceClass}">
                    <i class="fas fa-check-circle me-1"></i>
                    ${confidence} Güven
                </div>
                <div class="mt-3">
                    <small class="text-muted">
                        <i class="fas fa-info-circle me-1"></i>
                        Bu tahmin rehber niteliğindedir
                    </small>
                </div>
            </div>
        `;
        
        // Add success animation
        predictionResult.style.animation = 'fadeInUp 0.6s ease-out';
    }
    
    // Show error message
    function showError(errorMessage) {
        predictionResult.innerHTML = `
            <div class="text-center">
                <i class="fas fa-exclamation-triangle text-danger mb-3" style="font-size: 3rem;"></i>
                <h4 class="text-danger">Hata Oluştu</h4>
                <p class="text-muted">${errorMessage}</p>
                <button class="btn btn-outline-primary btn-sm" onclick="location.reload()">
                    <i class="fas fa-refresh me-1"></i>
                    Tekrar Dene
                </button>
            </div>
        `;
    }
    
    // Get confidence class based on confidence level
    function getConfidenceClass(confidence) {
        switch(confidence.toLowerCase()) {
            case 'yüksek':
                return 'confidence-high';
            case 'orta':
                return 'confidence-medium';
            case 'düşük':
                return 'confidence-low';
            default:
                return 'confidence-medium';
        }
    }
    
    // Add input validation and real-time feedback
    const inputs = form.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            validateInput(this);
        });
        
        input.addEventListener('blur', function() {
            validateInput(this);
        });
    });
    
    // Input validation function
    function validateInput(input) {
        const value = input.value;
        const min = input.getAttribute('min');
        const max = input.getAttribute('max');
        
        // Remove previous validation classes
        input.classList.remove('is-valid', 'is-invalid');
        
        if (value === '') {
            input.classList.add('is-invalid');
            return false;
        }
        
        if (min && parseFloat(value) < parseFloat(min)) {
            input.classList.add('is-invalid');
            return false;
        }
        
        if (max && parseFloat(value) > parseFloat(max)) {
            input.classList.add('is-invalid');
            return false;
        }
        
        input.classList.add('is-valid');
        return true;
    }
    
    // Add tooltips for better user experience
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add form reset functionality
    const resetButton = document.createElement('button');
    resetButton.type = 'button';
    resetButton.className = 'btn btn-outline-secondary btn-sm ms-2';
    resetButton.innerHTML = '<i class="fas fa-undo me-1"></i> Sıfırla';
    resetButton.addEventListener('click', function() {
        form.reset();
        predictionResult.innerHTML = `
            <div class="prediction-placeholder">
                <i class="fas fa-arrow-up text-primary mb-3" style="font-size: 3rem;"></i>
                <h4 class="text-muted">Tahmin Sonucu</h4>
                <p class="text-muted">Formu doldurun ve tahmin alın</p>
            </div>
        `;
        
        // Remove validation classes
        inputs.forEach(input => {
            input.classList.remove('is-valid', 'is-invalid');
        });
    });
    
    // Add reset button to form
    const submitButton = form.querySelector('button[type="submit"]');
    submitButton.parentNode.appendChild(resetButton);
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl+Enter to submit form
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            form.dispatchEvent(new Event('submit'));
        }
        
        // Escape to reset form
        if (e.key === 'Escape') {
            resetButton.click();
        }
    });
    
    // Add auto-save functionality (optional)
    let autoSaveTimeout;
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            clearTimeout(autoSaveTimeout);
            autoSaveTimeout = setTimeout(() => {
                // Save form data to localStorage
                const formData = new FormData(form);
                const data = {};
                for (let [key, value] of formData.entries()) {
                    data[key] = value;
                }
                localStorage.setItem('predictionFormData', JSON.stringify(data));
            }, 1000);
        });
    });
    
    // Load saved form data
    const savedData = localStorage.getItem('predictionFormData');
    if (savedData) {
        try {
            const data = JSON.parse(savedData);
            Object.keys(data).forEach(key => {
                const input = form.querySelector(`[name="${key}"]`);
                if (input) {
                    input.value = data[key];
                    validateInput(input);
                }
            });
        } catch (e) {
            console.error('Error loading saved form data:', e);
        }
    }
    
    // Add print functionality
    const printButton = document.createElement('button');
    printButton.type = 'button';
    printButton.className = 'btn btn-outline-info btn-sm ms-2';
    printButton.innerHTML = '<i class="fas fa-print me-1"></i> Yazdır';
    printButton.addEventListener('click', function() {
        window.print();
    });
    
    // Add print button to form
    submitButton.parentNode.appendChild(printButton);
    
    // Add copy result functionality
    function copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(function() {
            // Show success message
            const toast = document.createElement('div');
            toast.className = 'toast-notification';
            toast.innerHTML = `
                <div class="toast-content">
                    <i class="fas fa-check-circle text-success me-2"></i>
                    Sonuç panoya kopyalandı!
                </div>
            `;
            document.body.appendChild(toast);
            
            setTimeout(() => {
                toast.remove();
            }, 3000);
        });
    }
    
    // Add copy button to prediction result
    const copyButton = document.createElement('button');
    copyButton.type = 'button';
    copyButton.className = 'btn btn-outline-success btn-sm mt-2';
    copyButton.innerHTML = '<i class="fas fa-copy me-1"></i> Kopyala';
    copyButton.style.display = 'none';
    
    // Show copy button when prediction is made
    const originalShowPrediction = showPrediction;
    showPrediction = function(sampleSize, confidence) {
        originalShowPrediction(sampleSize, confidence);
        copyButton.style.display = 'inline-block';
        copyButton.onclick = () => copyToClipboard(`Örneklem Büyüklüğü: ${sampleSize} (${confidence} Güven)`);
        
        const resultContainer = predictionResult.querySelector('.prediction-result');
        if (resultContainer) {
            resultContainer.appendChild(copyButton);
        }
    };
});

// Add CSS for toast notifications
const style = document.createElement('style');
style.textContent = `
    .toast-notification {
        position: fixed;
        top: 20px;
        right: 20px;
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 15px 20px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        z-index: 9999;
        animation: slideInRight 0.3s ease-out;
    }
    
    .toast-content {
        display: flex;
        align-items: center;
        font-weight: 500;
    }
    
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @media print {
        .navbar, .btn, footer {
            display: none !important;
        }
        
        .hero-section {
            background: none !important;
            color: black !important;
        }
        
        .card {
            box-shadow: none !important;
            border: 1px solid #dee2e6 !important;
        }
    }
`;
document.head.appendChild(style);
