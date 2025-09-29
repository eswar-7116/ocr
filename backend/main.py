from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional
import uvicorn
import json # You need this for JSON decoding
from fastapi.middleware.cors import CORSMiddleware 

# CRITICAL FIX: Import the verification function
from ocr_service import extract_fields, process_registration_document, verify_data 

app = FastAPI(
    title="OCR and Data Verification API",
    description="API for extracting text from documents and verifying data."
)

# Add CORS middleware (REMAINS UNCHANGED)
origins = [
    "http://localhost",
    "http://localhost:5500", 
    "http://127.0.0.1",
    "http://127.0.0.1:5500", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schemas ---
# Removed VerificationRequest and VerificationResult schemas, 
# as the endpoint will return the dictionary directly from verify_data

class OCRResult(BaseModel):
    full_text: str
    extracted_fields: dict

class RegistrationExtractionResult(BaseModel):
    full_text: str
    extracted_fields: dict

@app.get("/")
async def Greeting():
    return "Hey, How are you"

# --- API 1: OCR Extraction (Image Only) ---
@app.post("/ocr/extract", response_model=OCRResult, summary="Extract text and fields from an image")
async def ocr_extraction_api(file: UploadFile = File(...), doc_type: str = Form("printed")):
    """
    API 1: Accepts a scanned image and extracts text and relevant fields using OCR.
    """
    if not file.content_type.startswith("image/"):
        print("Invalid file type.")
        raise HTTPException(status_code=400, detail="Invalid file type. Only image files are supported.")
    
    try:
        image_bytes = await file.read()
        result = process_registration_document(image_bytes, file.content_type, doc_type.lower())
        return OCRResult(full_text=result["full_text"], extracted_fields=result["extracted_fields"])
    except Exception as e:
        # Check if model loading failed and return 500 error
        if "OCR models were not initialized" in str(e):
             print(f"Models not initialized: {e}")
             raise HTTPException(status_code=500, detail="OCR models failed to load. Check server console.")
        print(f"OCR failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {e}")

# --- API 1 (Alternative): Registration Extraction (Multi-Type) ---
@app.post("/register/extract", response_model=RegistrationExtractionResult, summary="Extract registration data from various document types")
async def register_extraction_api(file: UploadFile = File(...), doc_type: str = Form("printed")):
    """
    API 1: Extracts registration data from document types including image, PDF, DOCX, TXT.
    """
    allowed_types = [
        "image/jpeg", "image/png", "image/gif", "image/bmp", "image/webp",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # .docx
        "text/plain"
    ]
    if file.content_type not in allowed_types:
        print("Unsupported file type.")
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Allowed types are: {', '.join(allowed_types)}")
    
    try:
        file_bytes = await file.read()
        result = process_registration_document(file_bytes, file.content_type, doc_type.lower())
        return RegistrationExtractionResult(full_text=result["full_text"], extracted_fields=result["extracted_fields"])
    except ValueError as ve:
        print(f"ValueError: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Document processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {e}")


# --- API 2: Data Verification (MANDATORY TASK) ---
@app.post("/ocr/verify", summary="Verify submitted data against fresh OCR extraction", response_model=dict)
async def data_verification_api(
    document: UploadFile = File(..., description="Original scanned document (image/PDF)."),
    submitted_data: str = Form(..., description="Submitted form fields (JSON string of key-value pairs)."),
    doc_type: str = Form("printed", description="Document type ('printed' or 'handwritten').")
):
    """
    API 2: Accepts submitted form data and the original document, 
    re-runs OCR, and compares values using fuzzy matching (confidence score).
    """
    try:
        # 1. Parse submitted JSON data
        submitted_data_dict = json.loads(submitted_data)
    except json.JSONDecodeError:
        print("Invalid JSON format for submitted_data.")
        raise HTTPException(status_code=400, detail="Invalid JSON format for submitted_data.")
    
    try:
        # 2. Re-run OCR Extraction to get the ground truth
        file_bytes = await document.read()
        
        extraction_result = process_registration_document(
            file_bytes, document.content_type, doc_type.lower()
        )
        extracted_data = extraction_result['extracted_fields']
        
        # 3. Perform Verification using the fuzzy matching logic
        # This function (verify_data) calculates the confidence score
        verification_results = verify_data(extracted_data, submitted_data_dict)
        
        return {
            "status": "verified",
            "verification_results": verification_results
        }
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Verification processing failed: {e}")


if __name__ == "__main__":
    # Starting the server on port 8000
    print("Server listening at port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)



# from fastapi import FastAPI, UploadFile, File, HTTPException, Form
# from pydantic import BaseModel
# from typing import Optional
# import uvicorn
# from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware

# from ocr_service import extract_fields, process_registration_document # Removed process_document_for_ocr

# app = FastAPI(
#     title="OCR and Data Verification API",
#     description="API for extracting text from documents and verifying data."
# )

# # Add CORS middleware
# origins = [
#     "http://localhost",
#     "http://localhost:5500", # Allow your frontend's origin
#     "http://127.0.0.1",
#     "http://127.0.0.1:5500", # Allow your frontend's origin
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class OCRResult(BaseModel):
#     full_text: str
#     extracted_fields: dict

# class RegistrationExtractionResult(BaseModel):
#     full_text: str
#     extracted_fields: dict

# class VerificationRequest(BaseModel):
#     extracted_data: dict
#     original_document_text: str # This would ideally be re-OCR'd or passed from extraction
#     # For simplicity, we'll assume original_document_text is the full text from the original OCR

# class VerificationResult(BaseModel):
#     field_name: str
#     submitted_value: Optional[str]
#     extracted_value: Optional[str]
#     match_status: str # "MATCH", "MISMATCH", "NOT_FOUND_SUBMITTED", "NOT_FOUND_EXTRACTED"
#     confidence_score: float = 1.0 # Placeholder for now

# @app.get("/")
# async def Greeting():
#     return "Hey, How are you"


# @app.post("/ocr/extract", response_model=OCRResult, summary="Extract text and fields from an image")
# async def ocr_extraction_api(file: UploadFile = File(...), doc_type: str = Form("printed")):
#     """
#     Accepts a scanned image (PNG, JPG) and extracts text and relevant fields using OCR.
#     Supports 'printed' and 'handwritten' document types.
#     """
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="Invalid file type. Only image files are supported.")
    
#     try:
#         image_bytes = await file.read()
#         # Use the consolidated process_registration_document for image OCR
#         result = process_registration_document(image_bytes, file.content_type, doc_type.lower())
#         return OCRResult(full_text=result["full_text"], extracted_fields=result["extracted_fields"])
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"OCR processing failed: {e}")

# @app.post("/register/extract", response_model=RegistrationExtractionResult, summary="Extract registration data from various document types")
# async def register_extraction_api(file: UploadFile = File(...), doc_type: str = Form("printed")):
#     """
#     Accepts a document (image, PDF, DOCX, TXT) and extracts registration-related fields.
#     Uses OCR for images and text parsing for other document types.
#     """
#     allowed_types = [
#         "image/jpeg", "image/png", "image/gif", "image/bmp", "image/webp",
#         "application/pdf",
#         "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # .docx
#         "text/plain"
#     ]
#     if file.content_type not in allowed_types:
#         raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Allowed types are: {', '.join(allowed_types)}")
    
#     try:
#         file_bytes = await file.read()
#         result = process_registration_document(file_bytes, file.content_type, doc_type.lower())
#         return RegistrationExtractionResult(full_text=result["full_text"], extracted_fields=result["extracted_fields"])
#     except ValueError as ve:
#         raise HTTPException(status_code=400, detail=str(ve))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Document processing failed: {e}")

# @app.post("/ocr/verify", response_model=list[VerificationResult], summary="Verify extracted data against submitted data")
# async def data_verification_api(request: VerificationRequest):
#     """
#     Compares submitted form data against text extracted from the original document.
#     Returns a list of verification results for each field.
#     """
#     results = []
    
#     # Re-extract fields from the original document text for comparison
#     # In a real scenario, you might re-OCR the original document or pass the full text from the extraction step
#     original_extracted_fields = extract_fields(request.original_document_text)
    
#     for field_name, submitted_value in request.extracted_data.items():
#         extracted_value = original_extracted_fields.get(field_name)
        
#         match_status = "UNKNOWN"
#         if submitted_value is None and extracted_value is None:
#             match_status = "NO_DATA"
#         elif submitted_value is None and extracted_value is not None:
#             match_status = "NOT_FOUND_SUBMITTED"
#         elif submitted_value is not None and extracted_value is None:
#             match_status = "NOT_FOUND_EXTRACTED"
#         elif submitted_value.strip().lower() == extracted_value.strip().lower():
#             match_status = "MATCH"
#         else:
#             match_status = "MISMATCH"
            
#         results.append(VerificationResult(
#             field_name=field_name,
#             submitted_value=submitted_value,
#             extracted_value=extracted_value,
#             match_status=match_status,
#             confidence_score=1.0 # Placeholder
#         ))
#     return results

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
