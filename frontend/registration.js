document.addEventListener('DOMContentLoaded', () => {
    const registrationUploadForm = document.getElementById('registrationUploadForm');
    const regDocumentUpload = document.getElementById('regDocumentUpload');
    const regDocType = document.getElementById('regDocType');
    const uploadStatus = document.getElementById('uploadStatus');

    const registrationForm = document.getElementById('registrationForm');
    const regName = document.getElementById('regName');
    const regAge = document.getElementById('regAge');
    const regGender = document.getElementById('regGender');
    const regAddress = document.getElementById('regAddress');
    const regEmail = document.getElementById('regEmail');
    const regPhone = document.getElementById('regPhone');

    const fieldStatusElements = {
        Name: document.getElementById('statusName'),
        Age: document.getElementById('statusAge'),
        Gender: document.getElementById('statusGender'),
        Address: document.getElementById('statusAddress'),
        Email: document.getElementById('statusEmail'),
        "Phone Number": document.getElementById('statusPhone'),
    };

    // Function to update form fields and status indicators
    function updateFormFields(extractedFields) {
        const fieldsToUpdate = {
            Name: regName,
            Age: regAge,
            Gender: regGender,
            Address: regAddress,
            Email: regEmail,
            "Phone Number": regPhone,
        };

        for (const key in fieldsToUpdate) {
            const inputElement = fieldsToUpdate[key];
            const statusElement = fieldStatusElements[key];
            const value = extractedFields[key];

            if (value) {
                inputElement.value = value;
                if (statusElement) {
                    statusElement.textContent = 'Filled';
                    statusElement.className = 'fill-status filled';
                }
            } else {
                inputElement.value = ''; // Clear if no value
                if (statusElement) {
                    statusElement.textContent = 'Unfilled';
                    statusElement.className = 'fill-status unfilled';
                }
            }
        }
    }

    // --- Registration Document Upload Form Submission ---
    registrationUploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        const file = regDocumentUpload.files[0];
        if (!file) {
            uploadStatus.textContent = 'Please select a document to upload.';
            uploadStatus.className = 'status-message error';
            return;
        }

        uploadStatus.textContent = 'Extracting data...';
        uploadStatus.className = 'status-message';

        const formData = new FormData();
        formData.append('file', file);
        // Only append doc_type if it's an image, backend will handle it
        if (file.type.startsWith('image/')) {
            formData.append('doc_type', regDocType.value);
        }

        try {
            const response = await fetch('http://localhost:8000/register/extract', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Document extraction failed.');
            }

            const result = await response.json();
            uploadStatus.textContent = 'Data extracted and form auto-filled!';
            uploadStatus.className = 'status-message success';
            updateFormFields(result.extracted_fields);

        } catch (error) {
            console.error('Error during document extraction:', error);
            uploadStatus.textContent = `Error: ${error.message}`;
            uploadStatus.className = 'status-message error';
        }
    });

    // --- Registration Form Submission (for actual registration) ---
    registrationForm.addEventListener('submit', (event) => {
        event.preventDefault();
        // Here you would typically send the form data to another backend endpoint
        // For this demo, we'll just log it.
        const formData = {
            name: regName.value,
            age: regAge.value,
            gender: regGender.value,
            address: regAddress.value,
            email: regEmail.value,
            phoneNumber: regPhone.value,
        };
        console.log('Registration Form Submitted:', formData);
        alert('Registration form submitted (check console for data).');
    });
});
