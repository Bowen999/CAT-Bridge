document.getElementById('file-uploader').addEventListener('click', function () {
  document.getElementById('upload1').click();
});

document.getElementById('file-uploader').addEventListener('dragover', function (e) {
  e.preventDefault();
  e.stopPropagation();
  e.dataTransfer.dropEffect = 'copy';
});

document.getElementById('file-uploader').addEventListener('drop', function (e) {
  e.preventDefault();
  e.stopPropagation();
  let files = e.dataTransfer.files;
  document.getElementById('upload1').files = files;
});
