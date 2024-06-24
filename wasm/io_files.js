var open_files_dialog_popup = function (extensions, multi) {
  var labelString = multi ? "Select Files" : "Select File";
  var { overlay, popup } = createOverlayPopup('show_browse_dialog', labelString, 400, 150, true, true);

  var file_selector_label = document.createElement('label');
  file_selector_label.setAttribute('for', 'FileSelectorTag');
  file_selector_label.setAttribute('style', 'position:absolute;top:50%;left:50%;transform:translate(-50%,50%);width: 120px;  height: 28px; border-radius: 4px;');
  file_selector_label.setAttribute('class', 'button');

  var file_selector_label_text = document.createElement('div');
  file_selector_label_text.innerHTML = "Browse...";
  file_selector_label_text.setAttribute('class', 'unselectable');
  file_selector_label_text.setAttribute('style', 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size: 14px;  font-weight: 600;  font-stretch: normal;  font-style: normal;  line-height: normal;  letter-spacing: normal;  text-align: center;  color: #fff;');

  var file_selector = document.createElement('input');
  file_selector.setAttribute('type', 'file');
  file_selector.setAttribute('id', 'FileSelectorTag');
  if (multi)
    file_selector.setAttribute('multiple', null);
  file_selector.setAttribute('onchange', 'open_files(event)');
  if (!is_ios())
    file_selector.setAttribute('accept', extensions);
  file_selector.setAttribute('style', 'display: none;');
  file_selector.setAttribute('align', 'center');

  file_selector_label.appendChild(file_selector_label_text);
  file_selector_label.appendChild(file_selector);

  popup.appendChild(file_selector_label);
  overlay.appendChild(popup);

  removeKeyboardEvents();
  document.body.appendChild(overlay);
}

var download_file_dialog_popup = function (defaultName, extensions) {
  var isLightThemeEnabled = getColorTheme();
  var { overlay, popup } = createOverlayPopup('show_download_dialog', "Save File", 440, 232);

  var textColor = isLightThemeEnabled ? '#181a1d' : '#fff';
  var bgColor = isLightThemeEnabled ? '#f5f6f9' : '#000';
  var name_label = document.createElement('label');
  name_label.setAttribute('style', 'width: 144px;height: 20px;position: absolute;top: 86px;left: 62px;margin-left: 0px;font-size: 14px;color:' + textColor );
  name_label.innerHTML = 'Name';
  name_label.setAttribute('class', 'unselectable');

  var name_selector = document.createElement('input');
  name_selector.setAttribute('type', 'text');
  name_selector.setAttribute('id', 'download_name');
  name_selector.setAttribute('style', 'position: absolute;top: 81px;left: 50%;transform: translate(-50%, 0px);background-color:' + bgColor +';border-radius: 4px;width: 198px;height: 26px;border: solid 1px #5f6369;color:' + textColor + 'padding: 0px 0px;');

  name_selector.value = defaultName;

  var ext_label = document.createElement('label');
  ext_label.setAttribute('style', 'width: 59px;height: 20px;font-size: 14px;position: absolute;color:' + textColor + ';top: 131px;left: 38px;');
  ext_label.innerHTML = 'Extension';
  ext_label.setAttribute('class', 'unselectable');

  var list_item = document.createElement('select');
  list_item.setAttribute('id', 'download_ext');
  list_item.setAttribute('style', 'position: absolute;top: 125px;left: 50%;transform: translate(-50%, 0px);background-color:' + bgColor + ';border-radius: 4px;width: 200px;height: 28px;border: solid 1px #5f6369;color:' + textColor + 'padding: 0px 0px;');
  var splitExt = extensions.split(', ');
  for (var i = 0; i < splitExt.length; i++) {
    var option_el = document.createElement('option');
    option_el.setAttribute('value', splitExt[i]);
    option_el.innerHTML = splitExt[i];
    option_el.setAttribute('class', 'unselectable');
    list_item.appendChild(option_el);
  }

  var btn_save = document.createElement('input');
  btn_save.setAttribute('type', 'button');
  btn_save.setAttribute('value', 'Save');
  btn_save.setAttribute('style', 'position: absolute;width: 100px;height: 28px;top: 194px;left: 50%;transform: translate(-50%, -50%);border-radius: 4px;color: #fff;font-size: 14px;font-weight: 600;border: none;');
  btn_save.setAttribute('class', 'button');
  btn_save.setAttribute('onclick', 'Module.ccall(\'emsSaveFile\', \'number\', [\'string\'], [document.getElementById(\'download_name\').value + document.getElementById(\'download_ext\').value]),addKeyboardEvents(),document.getElementById(\'show_download_dialog\').remove()');

  popup.appendChild(name_label);
  popup.appendChild(name_selector);
  popup.appendChild(ext_label);
  popup.appendChild(list_item);
  popup.appendChild(btn_save);

  removeKeyboardEvents();
  document.body.appendChild(overlay);
}

var open_directory_dialog_popup = function () {
  var { overlay, popup } = createOverlayPopup('show_browse_dialog', "Select Directory", 400, 150);  
  var file_selector_label = document.createElement('label');
  file_selector_label.setAttribute('for', 'FileSelectorTag');
  file_selector_label.setAttribute('style', 'position:absolute;top:50%;left:50%;transform:translate(-50%,50%);width: 120px;  height: 28px; border-radius: 4px;');
  file_selector_label.setAttribute('class', 'button');

  var file_selector_label_text = document.createElement('div');
  file_selector_label_text.innerHTML = "Browse...";
  file_selector_label_text.setAttribute('class', 'unselectable');
  file_selector_label_text.setAttribute('style', 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size: 14px;  font-weight: 600;  font-stretch: normal;  font-style: normal;  line-height: normal;  letter-spacing: normal;  text-align: center;  color: #fff;');

  var file_selector = document.createElement('input');
  file_selector.setAttribute('type', 'file');
  file_selector.setAttribute('id', 'FileSelectorTag');
  file_selector.setAttribute('directory', null);
  file_selector.setAttribute('mozdirectory', null);
  file_selector.setAttribute('webkitdirectory', null);
  file_selector.setAttribute('onchange', 'open_dir(event)');
  file_selector.setAttribute('style', 'display: none;');
  file_selector.setAttribute('align', 'center');

  file_selector_label.appendChild(file_selector_label_text);
  file_selector_label.appendChild(file_selector);

  popup.appendChild(file_selector_label);
  overlay.appendChild(popup);

  removeKeyboardEvents();
  document.body.appendChild(overlay);
}

var open_files = function (e) {
  if (!GLFW.active) {
    addKeyboardEvents();
    document.getElementById('show_browse_dialog').remove();
    return;
  }
  if (!e.target || !e.target.files || e.target.files.length == 0) {
    addKeyboardEvents();
    document.getElementById('show_browse_dialog').remove();
    return;
  }
  e.preventDefault();
  var filenames = _malloc(e.target.files.length * 4);
  var filenamesArray = [];
  var count = e.target.files.length;
  var written = 0;
  var drop_dir = ".use_open_files";
  FS.createPath("/", drop_dir);
  function save(file) {
    var path = "/" + drop_dir + "/" + file.name.replace(/\//g, "_");
    var reader = new FileReader();
    reader.onloadend = e => {
      if (reader.readyState != 2) {
        ++written;
        out("failed to read opened file: " + file.name + ": " + reader.error);
        return;
      }
      var data = e.target.result;
      FS.writeFile(path, new Uint8Array(data));
      if (++written === count) {
        Module.ccall('emsOpenFiles', 'number', ['number', 'Int8Array'], [count, filenames]);
        for (var i = 0; i < filenamesArray.length; ++i) {
          _free(filenamesArray[i]);
        }
        _free(filenames);
      }
      // enforce several frames to toggle animation when popup closed
      for (var i = 0; i < 500; i += 100)
        setTimeout(function () { Module.ccall('emsPostEmptyEvent', 'void', ['number'], [1]); }, i);
    };
    reader.readAsArrayBuffer(file);
    var filename = stringToNewUTF8(path);
    filenamesArray.push(filename);
    if (typeof GROWABLE_HEAP_U32 !== 'undefined')
      GROWABLE_HEAP_U32()[filenames + i * 4 >> 2] = filename;
    else
      HEAP32[filenames + i * 4 >> 2] = filename;
  }
  for (var i = 0; i < count; ++i) {
    save(e.target.files[i]);
  }
  addKeyboardEvents();
  document.getElementById('show_browse_dialog').remove();
  return false;
};

var prevSize = 0;
var save_file = function (filename) {
  var checkPath = function (filename) {
    if (!FS.analyzePath(filename).exists) {
      setTimeout(() => {
        checkPath(filename);
      }, 0);
      return;
    }
    var size = FS.stat(filename).size;
    if (size === 0 || size !== prevSize) {
      prevSize = size;
      setTimeout(() => {
        checkPath(filename);
      }, 0);
      return;
    }
    let content = FS.readFile(filename);

    var a = document.createElement('a');
    a.download = filename;
    var mime = "application/octet-stream";
    a.href = URL.createObjectURL(new Blob([content], { type: mime }));
    a.style.display = 'none';

    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(a.href);
    }, 0);
  };
  prevSize = 0;
  checkPath(filename);
};

var open_dir = function (e) {
  if (!GLFW.active) {
    addKeyboardEvents();
    document.getElementById('show_browse_dialog').remove();
    return;
  }
  if (!e.target || !e.target.files || e.target.files.length == 0) {
    addKeyboardEvents();
    document.getElementById('show_browse_dialog').remove();
    return;
  }
  e.preventDefault();

  var drop_dir = ".use_open_files";
  FS.createPath("/", drop_dir);

  var root_dir_name = "";
  var written = 0;
  const files = e.target.files;
  function save(file) {
    const file_path = file.webkitRelativePath;
    const dir_path = file_path.substring(0, file_path.lastIndexOf('/'));
    if (root_dir_name === "") {
      root_dir_name = file_path.substring(0, file_path.indexOf('/'));
    }

    var reader = new FileReader();
    reader.onloadend = e => {
      if (reader.readyState != 2) {
        ++written;
        out("failed to read opened file: " + file.name + ": " + reader.error);
        return;
      }

      var data = e.target.result;
      FS.createPath("/" + drop_dir, dir_path);
      FS.writeFile(`/${drop_dir}/${file_path}`, new Uint8Array(data));

      if (++written === files.length) {
        Module.ccall('emsOpenDirectory', 'number', ['string'], [`/${drop_dir}/${root_dir_name}`]);
      }

      // enforce several frames to toggle animation when popup closed
      for (var i = 0; i < 500; i += 100)
        setTimeout(function () { Module.ccall('emsPostEmptyEvent', 'void', ['number'], [1]); }, i);
    };
    reader.readAsArrayBuffer(file);
  }
  for (const file of files) {
    save(file);
  }

  addKeyboardEvents();
  document.getElementById('show_browse_dialog').remove();
  return false;
};
