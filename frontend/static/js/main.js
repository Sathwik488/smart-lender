// Smart Lender - Frontend Scripts

document.addEventListener('DOMContentLoaded', function () {
    // 1. Prediction Form dynamic calculation and loading trigger
    const predictionForm = document.getElementById('predictionForm');
    const loadingOverlay = document.getElementById('loadingOverlay');

    if (predictionForm) {
        const appIncomeInput = document.getElementById('applicant_income');
        const coappIncomeInput = document.getElementById('coapplicant_income');
        const loanAmountInput = document.getElementById('loan_amount');
        const loanTermInput = document.getElementById('loan_amount_term');

        const totalIncomePreview = document.getElementById('previewTotalIncome');
        const emiPreview = document.getElementById('previewEMI');

        function updatePreviews() {
            const appIncome = parseFloat(appIncomeInput?.value) || 0;
            const coappIncome = parseFloat(coappIncomeInput?.value) || 0;
            const loanAmountK = parseFloat(loanAmountInput?.value) || 0;
            const term = parseFloat(loanTermInput?.value) || 360;

            const total = appIncome + coappIncome;
            if (totalIncomePreview) {
                totalIncomePreview.textContent = '$' + total.toLocaleString('en-US', { minimumFractionDigits: 0 });
            }

            if (emiPreview && term > 0 && loanAmountK > 0) {
                const principal = loanAmountK * 1000;
                const emi = (principal / term);
                emiPreview.textContent = '$' + emi.toLocaleString('en-US', { maximumFractionDigits: 0 }) + ' / mo';
            }
        }

        [appIncomeInput, coappIncomeInput, loanAmountInput, loanTermInput].forEach(el => {
            if (el) {
                el.addEventListener('input', updatePreviews);
                el.addEventListener('change', updatePreviews);
            }
        });
        updatePreviews();

        predictionForm.addEventListener('submit', function (e) {
            if (!predictionForm.checkValidity()) {
                e.preventDefault();
                e.stopPropagation();
                predictionForm.classList.add('was-validated');
                return;
            }
            if (loadingOverlay) {
                loadingOverlay.style.display = 'flex';
            }
        });
    }

    // 2. History table row filter and modal inspection
    const historySearch = document.getElementById('historySearch');
    const historyTable = document.getElementById('historyTable');

    if (historySearch && historyTable) {
        historySearch.addEventListener('keyup', function () {
            const query = historySearch.value.toLowerCase();
            const rows = historyTable.querySelectorAll('tbody tr');
            rows.forEach(row => {
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(query) ? '' : 'none';
            });
        });
    }
});
