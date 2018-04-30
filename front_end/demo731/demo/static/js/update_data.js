$(document).ready(function () {
  window.setInterval(function(){
     $.get("/refresh")
          .done(function (data) {
          update(data);
      }).fail(function(jqXHR, textStatus, errorThrown) {
            console.log(errorThrown);
      });
}, 1000);
});

function update(data) {
  // Process posts
  var table = document.getElementById("myTable")
  var inner_html = "";
  inner_html += "<tr>\n";
  inner_html += "<th scope=\"col\">MAC address</th>\n";
  inner_html += "<th scope=\"col\">Device type</th>\n";
  inner_html += "</tr>\n";
  for(var i = 0; i < data.mac_devices.length; i++) {
    inner_html += "<tr>\n";
    inner_html += "<td>" + data.mac_devices[i].mac_addr + "</td>\n";
    inner_html += "<td>" + data.mac_devices[i].device_type + "</td>\n";
    inner_html += "</tr>\n";
  }
  table.innerHTML = inner_html;
}


