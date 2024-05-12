<!doctype html>
<html lang="en">
  <head>
    <title>Title</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
<style>
    body{
        font-family: 'Roboto', sans-serif;
    }
    .nav-back{
        background: rgba(17, 219, 207,0.8);
        color: #fff;
        height: 10vh;
    }
    .navbar-brand{
        color: #000;
    }
    .navbar-brand:hover{
        color: #fff;
    }
    .nav-link{
        color: #000;
    }
    .nav-link:hover{
        color:#fff;
    }
    .navbar-toggler{
        outline: none!important;
    }
    #hero {
        width: 100%;
        height: 45vh;
        background-size: cover;
        position: relative;
        }
        
        #hero .container {
        padding-top: 50px;
        }
        
        #hero:before {
        content: "";
        background: rgba(0, 0, 0, 0.6);
        position: absolute;
        bottom: 0;
        top: 0;
        left: 0;
        right: 0;
        }
        
        #hero h1 {
        margin: 0 0 10px 0;
        font-size: 48px;
        font-weight: 700;
        line-height: 56px;
        color: #fff;
        }
        
        #hero h2 {
        color: #eee;
        margin-bottom: 40px;
        font-size: 15px;
        font-weight: 500;
        font-family: "Open Sans", sans-serif;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        }
        
        .main-btn {
        font-family: "Poppins", sans-serif;
        text-transform: uppercase;
        font-weight: 500;
        font-size: 14px;
        letter-spacing: 1px;
        display: inline-block;
        padding: 8px 28px;
        border-radius: 50px;
        transition: 0.5s;
        margin: 10px;
        border: 2px solid #fff;
        color: #fff;
        }
        
        .main-btn:hover,.main2-btn:hover {
        background: #11dbcf;
        border: 2px solid #11dbcf;
        text-decoration: none;
        color:#080505;
        }
        .main2-btn {
        font-family: "Poppins", sans-serif;
        text-transform: uppercase;
        font-weight: 500;
        font-size: 14px;
        letter-spacing: 1px;
        display: inline-block;
        padding: 8px 28px;
        border-radius: 50px;
        transition: 0.5s;
        margin: 10px;
        border: 2px solid #180f0f;
        color: #0a0707;
        }
    .service {
        text-align: center;
        padding: 20px 20px 20px 20px;
        transition: all ease-in-out 0.3s;
        background: rgba(17, 219, 207,0.8);
        box-shadow: 0px 5px 90px 0px rgba(3, 117, 111, 0.63);
        width: 100%;
        }
       
    .service .icon {
        margin: 0 auto;
        width: 64px;
        height: 64px;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: ease-in-out 0.3s;
        position: relative;
        background: #f5f5f5;
        border-radius: 50px;
        }
        
        .service .icon i {
        font-size: 32px;
        transition: 0.5s;
        line-height: 1;
        position: relative;
        }
        
        .service h4 {
        font-weight: 600;
        margin: 10px 0 15px 0;
        font-size: 22px;
        }
        
        .service p {
        line-height: 24px;
        font-size: 14px;
        margin-bottom: 0;
        }
        
        .services i{
        color: #47aeff;
        }

        .bg-faded {
        background-color: rgba(17, 219, 207,0.8);
        }
        
        @media (min-width: 992px) {
        .about .about-img {
            width: 75%;
            float: right;
        }
        }
        @media (min-width: 1200px) {
        .about .about-text {
            width: 45%;
        }
      }
</style>
<script src="https://kit.fontawesome.com/332a215f17.js" crossorigin="anonymous"></script>

    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
  </head>
  <body>
    <nav class="navbar navbar-expand-lg nav-back fixed-top" 
    id="mainNav">
      <div class="container">
          <a class="navbar-brand" href="#">Oral Cancer Prediction and Risk Analysis</a>
          <button class="navbar-toggler navbar-toggler-right" type="button"
           data-toggle="collapse" data-target="#navbarResponsive" 
           aria-controls="navbarResponsive" aria-expanded="false" 
           aria-label="Toggle navigation"><i class="fas fa-syringe fa-2x"></i>
           </button>
          <div class="collapse navbar-collapse" id="navbarResponsive">
              <ul class="navbar-nav ml-auto">
                <li class="nav-item"><a class="nav-link" href="#about">About Oral Cancer</a></li>
                <li class="nav-item"><a class="nav-link" href="causes.html">Causes</a></li>
                <li class="nav-item"><a class="nav-link" href="#treatment.html">Treatment</a></li>
              </ul>
          </div>
      </div>
  </nav>

   <section id="hero" class="d-flex align-items-center">
    <div class="container text-center position-relative">
      <h1>Unmask Oral Cancer Before It Takes Hold!</h1>
      <h2>Early Detection, Brighter Smiles</h2>
      <a href="symptoms.html" class="main-btn">Get To Know the Symptoms Of oral Cancer</a>
    </div>
  </section>
  <!-- End Hero -->

   <section id="services" class="services mt-5 mb-3 py-3">
    <div class="container">
        <div class="section-title">
            <h1 class="text-center">One Stop Solution to Detect Oral cancer at the Earliest</h2>
                <br>
                <br>
        </div>  
        <div class="row">
            <div class="col-lg-6 col-md-6">
                <div class="service shadow">
                    <div class="icon">
                        <i class="fas fa-tooth"></i>
                    </div>
                    <h2>Clinical Data Analysis :</h2>
                    <form method="post" action="/" enctype="multipart/form-data">
                        <h4>Update Clinical Details here</h4>
                          <div class="form-part-1">
                            <label for="localization">Localization:</label>
                            <select name="localization" required>
                                <option value="2">Gingiva</option>
                                <option value="4">Palate</option>
                                <option value="5">Tongue</option>
                                <option value="3">Lip</option>
                                <option value="1">Floor of mouth</option>
                                <option value="0">Buccal mucosa</option>
                            </select><br>

                            <label>Gender:</label><br>
                            <input type="radio" id="female" name="gender" value="0" required>
                            <label for="female">Female</label><br>
                            <input type="radio" id="male" name="gender" value="1" required>
                            <label for="male">Male</label><br>
                    
                            <label for="tobacco_use">Tobacco Use:</label>
                            <select name="tobacco_use" required>
                                <option value="3">Yes</option>
                                <option value="0">Former</option>
                                <option value="1">No</option>
                            </select><br>
                    
                            <label for="alcohol_consumption">Alcohol Consumption:</label>
                            <select name="alcohol_consumption" required>
                                <option value="3">Yes</option>
                                <option value="0">Former</option>
                                <option value="1">No</option>
                            </select><br>
                    
                            <label for="sun_exposure">Sun Exposure (for more than 8 hours):</label>
                            <select name="sun_exposure" required>
                                <option value="2">Yes</option>
                                <option value="0">No</option>
                            </select><br>
                            <label for="size">Lesion size:</label>
                            <input type="numbers" name="size" required><br>
                            <label>Age group:</label><br>
                            <input type="radio" id="age_group_1" name="age_group" value="0" required>
                            <label for="age_group_1">&lt40</label><br>
                            <input type="radio" id="age_group_2" name="age_group" value="1">
                            <label for="age_group_2">40-60</label><br>
                            <input type="radio" id="age_group_3" name="age_group" value="2">
                            <label for="age_group_3">&gt60</label>
                            <br>
                          </div>
                      
                </div>
            </div>
            <div class="col-lg-6 col-md-6">
                <div class="service shadow">
                <div class="icon">
                    <i class="fas fa-syringe fa-2x"></i>
                </div>
                <br>
                <label for="district">Select your district:</label>
            <select id="district" name="district">
            <option value="CHENGALPATTU">Chengalpattu</option>
            <option value="CHENNAI">Chennai</option>
            <option value="COIMBATORE">Coimbatore</option>
            <option value="CUDDALORE">Cuddalore</option>
            <option value="DHAHRMAPURI">Dharmapuri</option>
            <option value="DINDIGUL">Dindigul</option>
            <option value="ERODE">Erode</option>
            <option value="KALLAKURICHI">Kallakurichi</option>
            <option value="KANCHEEPURAM">Kanchipuram</option>
            <option value="KANYAKUMARI">Kanniyakumari</option>
            <option value="KARUR">Karur</option>
            <option value="KRISHNAGIRI">Krishnagiri</option>
            <option value="MADURAI">Madurai</option>
            <option value="MAYILADUTHURAI">Mayiladuthurai</option>
            <option value="NAMAKKAL">Namakkal</option>
            <option value="PERAMBALUR">Perambalur</option>
            <option value="PUDUKOTTAI">Pudukkottai</option>
            <option value="RAMANATHAPURAM">Ramanathapuram</option>
            <option value="RANIPET">Ranipet</option>
            <option value="SALEM">Salem</option>
            <option value="SIVAGANGAI">Sivagangai</option>
            <option value="TENKASI">Tenkasi</option>
            <option value="THANJAVUR">Thanjavur</option>
            <option value="THENI">Theni</option>
            <option value="THIRUVALLUR">Thiruvallur</option>
            <option value="TIRUVARUR">Thiruvarur</option>
            <option value="TIRUCHIRAPPALLI">Tiruchirappalli</option>
            <option value="TIRUNELVELI">Tirunelveli</option>
            <option value="TIRUPUR">Tiruppur</option>
            <option value="TIRUVANNAMALAI">Tiruvannamalai</option>
            <option value="TUTICORIN">Tuticorin</option>
            <option value="VELLORE">Vellore</option>
            <option value="VILLUPURAM">Viluppuram</option>
            <option value="VIRUDHUNAGAR">Virudhunagar</option>
            </select>
            <br>
                <br>
                <input class="main2-btn" type="submit" value="Predict">
                <br>
                <br>
                <br>
                    </form>
                </div>
            </div>
        </div>
        
    </div>
  </section>
   
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js" integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>
  </body>
</html>