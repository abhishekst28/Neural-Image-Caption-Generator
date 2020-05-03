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

    function playSound(url){
  var audio = document.createElement('audio');
  audio.style.display = "none";
  audio.src = url;
  audio.autoplay = true;
  audio.onended = function(){
    audio.remove() //Remove when played.
  };
  document.body.appendChild(audio);
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
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(' Result : ' + data);
                var k=data.search("delimits");
                var text_style=data.substring(0,k);
                var audio_style=data.substring(k+9);
                //document.getElementById("demo1").innerHTML = text_style;
                //document.getElementById("demo2").innerHTML = audio_style;
                console.log(text_style);
                console.log(audio_style);
                $('#result').text(' Result :  ' + text_style);
                //var music= new Audio(audio_style)
                //music.play();
                //console.log("once");

                //var music2= new Audio("C:/Users/toshn/Desktop/mits/example.mp3")
                //music2.play();
                //console.log("tonce");
                //playSound(C:/Users/toshn/Desktop/mits/example.mp3);
                //playSound("C:/Users/toshn/Desktop/mits/example.mp3");
                //console.log("3once");

                console.log('Success!');
            },
        });
    });

});
