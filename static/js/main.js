$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // data = {prediction, tumor_probability, threshold, message}
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text('Result: ' + data.message +
                    ' (P(tumour) = ' + (data.tumor_probability * 100).toFixed(1) + '%)');
            },
            error: function (xhr) {
                $('.loader').hide();
                $('#btn-predict').show();
                var msg = (xhr.responseJSON && xhr.responseJSON.error) || 'Prediction failed';
                $('#result').fadeIn(600).text('Error: ' + msg);
            },
        });
    });

});