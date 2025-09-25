document.addEventListener('DOMContentLoaded', () => {
    const ocrExtractForm = document.getElementById('ocrExtractForm');
    const imageUpload = document.getElementById('imageUpload');
    const docType = document.getElementById('docType');
    const ocrFullText = document.getElementById('ocrFullText');
    const ocrExtractedFields = document.getElementById('ocrExtractedFields');

    const dataVerifyForm = document.getElementById('dataVerifyForm');
    const submittedName = document.getElementById('submittedName');
    const submittedDOB = document.getElementById('submittedDOB');
    const submittedID = document.getElementById('submittedID');
    const submittedAddress = document.getElementById('submittedAddress');
    const originalDocumentText = document.getElementById('originalDocumentText');
    const verifyOutput = document.getElementById('verifyOutput');

    // --- OCR Extraction Form Submission ---
    ocrExtractForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        const file = imageUpload.files[0];
        if (!file) {
            alert('Please select an image file.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('doc_type', docType.value);

        ocrFullText.textContent = 'Processing...';
        ocrExtractedFields.textContent = 'Processing...';

        try {
            const response = await fetch('http://localhost:8000/ocr/extract', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'OCR extraction failed.');
            }

            const result = await response.json();
            ocrFullText.textContent = result.full_text;
            ocrExtractedFields.textContent = JSON.stringify(result.extracted_fields, null, 2);
            originalDocumentText.value = result.full_text; // Auto-fill for verification

        } catch (error) {
            console.error('Error during OCR extraction:', error);
            ocrFullText.textContent = `Error: ${error.message}`;
            ocrExtractedFields.textContent = `Error: ${error.message}`;
        }
    });

    // --- Data Verification Form Submission ---
    dataVerifyForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        const submittedData = {
            "Name": submittedName.value || null,
            "DOB": submittedDOB.value || null,
            "ID Number": submittedID.value || null,
            "Address": submittedAddress.value || null,
        };

        const requestBody = {
            extracted_data: submittedData,
            original_document_text: originalDocumentText.value,
        };

        if (!originalDocumentText.value) {
            alert('Please perform OCR extraction first or provide original document text.');
            return;
        }

        verifyOutput.textContent = 'Verifying...';

        try {
            const response = await fetch('http://localhost:8000/ocr/verify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Data verification failed.');
            }

            const results = await response.json();
            let outputHtml = '';
            results.forEach(item => {
                const statusClass = `match-status-${item.match_status}`;
                outputHtml += `<div>
                    <strong>${item.field_name}:</strong>
                    <span class="${statusClass}">${item.match_status}</span><br>
                    Submitted: ${item.submitted_value || 'N/A'}<br>
                    Extracted: ${item.extracted_value || 'N/A'}
                </div><br>`;
            });
            verifyOutput.innerHTML = outputHtml;

        } catch (error) {
            console.error('Error during data verification:', error);
            verifyOutput.textContent = `Error: ${error.message}`;
        }
    });
});
