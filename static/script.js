
let currentXhr = null;

function uploadFile() {

    const fileInput = document.getElementById("fileInput");
    const labelInput = document.querySelector("input[name='label']");
    const progressBar = document.getElementById("progressBar");
    const progressText = document.getElementById("progressText");
    const progressContainer = document.getElementById("progressContainer");
    const cancelBtn = document.getElementById("cancelBtn");

    const analyzeBtn = document.getElementById("analyzeBtn");
    const btnText = document.getElementById("btnText");

    const file = fileInput.files[0];
    const label = labelInput.value.trim();

    // Validation
    if (!label) {
        showToast("Image title is required", "error");
        return;
    }

    if (!file) {
        showToast("Please select an image", "error");
        return;
    }

    const allowedTypes = ["image/png","image/jpeg","image/jpg"]

    if (!allowedTypes.includes(file.type)) {
        showToast("Only PNG, JPG, JPEG formats allowed","error")
        return
    }

    if (file.size > 5 * 1024 * 1024) {
        showToast("Image must be less than 5MB", "error");
        return;
    
    }

    // Button state
    analyzeBtn.disabled = true;
    analyzeBtn.classList.add("opacity-70","cursor-not-allowed");
    btnText.textContent = "Processing...";

    progressContainer.classList.remove("hidden");
    progressText.classList.remove("hidden");
    cancelBtn.classList.remove("hidden");

    progressBar.style.width = "0%";
    progressText.textContent = "Uploading: 0%";

    const formData = new FormData();
    formData.append("image", file);
    formData.append("label", label);

    currentXhr = new XMLHttpRequest();
    currentXhr.open("POST", "/upload", true);

    currentXhr.upload.onprogress = function (event) {
        if (event.lengthComputable) {
            let percent = Math.round((event.loaded / event.total) * 100);
            progressBar.style.width = percent + "%";
            progressText.textContent = "Uploading: " + percent + "%";
        }
    };

    currentXhr.onload = function () {

    cancelBtn.classList.add("hidden");

    if (currentXhr.status === 200) {

        let response = JSON.parse(currentXhr.responseText);

        if(response.status === "duplicate"){

            openDuplicateModal();
            resetButton();
            return;

        }else if(response.status === "success"){

            showToast("Image uploaded successfully", "success");

            progressBar.classList.remove("bg-indigo-500");
            progressBar.classList.add("bg-green-500");
            progressText.textContent = "Analyzing with AI...";

            setTimeout(function () {
                window.location.href = "/dashboard";
            }, 1500);
        }

    } else {
        showToast("Upload failed", "error");
        resetButton();
    }
};

    currentXhr.onerror = function () {
        showToast("Network error", "error");
        cancelBtn.classList.add("hidden");
        resetButton();
    };

    cancelBtn.addEventListener("click", function () {
        if (currentXhr) {
            currentXhr.abort();
            progressBar.style.width = "0%";
            progressText.textContent = "Upload Cancelled ❌";
            cancelBtn.classList.add("hidden");
            resetButton();
        }
    });

    currentXhr.send(formData);
}

function resetButton(){
    const analyzeBtn = document.getElementById("analyzeBtn");
    const btnText = document.getElementById("btnText");
    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove("opacity-70","cursor-not-allowed");
    btnText.textContent = "Analyze Image";
}

