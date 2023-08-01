let image_input = document.querySelector('#imageinput');
let imageEdges = document.querySelector('#imageEdges');
let edgedetect_method=document.querySelector("#edgedetectmethod");
let submit=document.querySelector("#submit_btn");
let image_data = ""




image_input.addEventListener('change', e => {
    if (e.target.files.length) {
      const reader = new FileReader();
      reader.onload = e => {
        if (e.target.result) {
          let img = document.createElement('img');
          img.id ="imageEdges";
          img.src = e.target.result;
          imageEdges.innerHTML = '';
          imageEdges.appendChild(img)
          image_data = e.target.result
  
          
        }
      };
      reader.readAsDataURL(e.target.files[0]);
    }
  });




submit.addEventListener('click', e => {
e.preventDefault();
send();
}
)

    
function send(){
        
      let formData = new FormData();

      try {
        // console.log("done1");

       if (image_data == "") {
        throw "error : not enought images "
      }
      // console.log("done2");
      formData.append('image_data',image_data);
      // formData.append("edgedetectmethod" ,edgedetect_method.value);
      // console.log("done3");
    
      console.log("formdata done")
      $.ajax({
        type: 'POST',
        url: '/facedetection',
        data: formData,
        cache: false,
        contentType: false,
        processData: false,
        async: true,
        success: function (backEndData) {
          console.log("done done done 1");
          var responce = JSON.parse(backEndData)
          console.log(responce)
          
          let ApplyEdges = document.getElementById("ApplyEdges")
          let info1 = document.getElementById("info1")

          ApplyEdges.remove()
          info1.remove()
          
          ApplyEdges = document.createElement("div")
          info1 = document.createElement("div")

          ApplyEdges.id = "ApplyEdges"
          info1.id = "info1"
          
          ApplyEdges.innerHTML = responce[1]
          info1.innerHTML = responce[2]
          
          let col2 = document.getElementById("Col2")
          col2.appendChild(ApplyEdges)

          let mycalc = document.getElementById("mycalc")
          mycalc.appendChild(info1)
          
          console.log("done done done 2");
        }
      })
      console.log("ajax done")
      console.log("Please Just Wait A Few Moments....")
      
    }
     catch (error) {
      console.log("please upload the image")
    } 
  }