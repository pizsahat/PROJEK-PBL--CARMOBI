//MASSAGE REGISTER
$(document).ready(function () {
  const urlParams = new URLSearchParams(window.location.search);
  const registerSuccess = urlParams.has("register_success");

  if (registerSuccess) {
    $(".container").prepend(`
            <div class="alert alert-success alert-dismissible fade show mt-3" role="alert">
              Register successful!
              <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
          `);
  }
});

//MASSAGE LOGIN
$(document).ready(function () {
  const urlParams = new URLSearchParams(window.location.search);
  const loginSuccess = urlParams.has("login_success");

  if (loginSuccess) {
    $(".container").prepend(`
                    <div class="alert alert-success alert-dismissible fade show mt-3" role="alert">
                      Login successful!
                      <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    </div>
                  `);
  }
});

//MASSAGE RESET PASSWORD
$(document).ready(function () {
  // Ambil parameter URL untuk menentukan apakah ada pesan sukses atau gagal
  const urlParams = new URLSearchParams(window.location.search);
  const resetSuccess = urlParams.has("reset_success");
  const errorMessage = urlParams.get("error_message");

  // Jika reset password berhasil, tampilkan alert success
  if (resetSuccess) {
    $(".container").prepend(`
                <div class="alert alert-success alert-dismissible fade show mt-3" role="alert">
                  Password reset email sent. Check your email inbox.
                  <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                </div>
              `);
  }

  // Jika ada error dalam reset password, tampilkan alert danger
  if (errorMessage) {
    $(".container").prepend(`
                <div class="alert alert-danger alert-dismissible fade show mt-3" role="alert">
                  Failed to send password reset email: ${errorMessage}
                  <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                </div>
              `);
  }
});

function kmPerLToMpg(kmPerL) {
  const conversionFactor = 2.35215;
  return kmPerL * conversionFactor;
}

$(document).ready(function () {
  // Fungsi untuk mendapatkan kuota prediksi
  function getPredictionQuota() {
    $.ajax({
      url: "/get_quota", // Endpoint untuk mendapatkan kuota prediksi
      type: "GET",
      success: function (response) {
        $("#prediction-quota").text(response.quota);
      },
      error: function (error) {
        console.error("Error:", error);
      },
    });
  }

  // Memanggil fungsi untuk mendapatkan kuota prediksi saat halaman dimuat
  getPredictionQuota();

  $("#prediction-form").on("submit", function (event) {
    event.preventDefault();

    let currentYear = new Date().getFullYear();
    let tahun = parseFloat($("#tahun").val());
    let isValid = true;
    $("#prediction-form input, #prediction-form select").each(function () {
      if ($(this).val() === "") {
        isValid = false;
        $(this).addClass("is-invalid");
      } else {
        $(this).removeClass("is-invalid");
      }
    });

    if (!isValid) {
      alert("Mohon isi semua bidang.");
      return;
    }

    if (tahun > currentYear) {
      alert(`Tahun tidak boleh lebih dari tahun ${currentYear}`);
      $("#tahun").addClass("is-invalid");
      return;
    } else {
      $("#tahun").removeClass("is-invalid");
    }

    let kilometer = parseFloat($("#kilometer").val());
    let pajak = parseFloat($("#pajak").val());
    let kmPerL = parseFloat($("#kmPerL").val());
    mpg = kmPerLToMpg(kmPerL);
    let cc = parseFloat($("#cc").val());
    let transmisi = parseFloat($("#transmisi").val());

    let bahan_bakar = $("#bahan_bakar").val().split(",").map(Number);
    let model = $("#model").val().split(",").map(Number);

    let features = [tahun, pajak, mpg, cc, transmisi]
      .concat(bahan_bakar)
      .concat(model)
      .concat([Math.log1p(kilometer)]);

    $.ajax({
      url: "/predict",
      type: "POST",
      contentType: "application/json",
      data: JSON.stringify({ features: features }),
      success: function (response) {
        let prediction = response.prediction[0];

        if (prediction < 10000000) {
          $("#result").text(
            "Terjadi kesalahan dalam prediksi. Nilai prediksi tidak valid."
          );
          return;
        }

        let rupiah = new Intl.NumberFormat("id-ID", {
          style: "currency",
          currency: "IDR",
        }).format(prediction);

        $("#result").text("Prediksi Harga: " + rupiah);

        // Setelah prediksi sukses, perbarui kuota prediksi
        getPredictionQuota();
      },
      error: function (response) {
        if (response.status === 403) {
          $("#result").text("");
          $("#error-message").text(
            "Silakan login untuk melanjutkan prediksi lebih lanjut."
          );
          $("#error-message").show();
        } else if (response.status === 400) {
          $("#result").text(""); // Hapus hasil prediksi sebelumnya
          $("#error-message").text(
            "Kuota prediksi sudah habis. Silakan coba lagi nanti."
          );
          $("#error-message").show();
        } else {
          console.error("Error:", response);
        }
      },
    });
  });
});

// Fungsi untuk memformat harga
function formatPrice(price) {
  return price.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ".");
}

// Fungsi untuk memformat input saat pengguna mengetik
function formatPriceInput(input) {
  let value = input.value.replace(/\D/g, ""); // Hanya angka
  let formattedValue = formatPrice(value);
  input.value = formattedValue;
}
