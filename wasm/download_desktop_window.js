var showDownloadWindow = function() {
    var { overlay, popup } = createOverlayPopup('show_download_window', "", 500, 300, true, true);
    var logo = document.createElement('img');
    logo.setAttribute('src', 'tool_not_supp.svg');
    logo.setAttribute('style', 'position: absolute;top: 0%;left: 50%;width:72px;height:72px;transform:translate(-50%,28px);');
    popup.appendChild( logo );

    var isLightThemeEnabled = getColorTheme();
    var captionColor = isLightThemeEnabled ? '#30343a' : '#fff';

    var caption = document.createElement('span');
    caption.setAttribute('class', 'unselectable');
    caption.innerHTML = "This tool is not supported by browser";
    caption.setAttribute('style', 'font-family: SegoeUIVariable-Display;font-size: 20px;  text-align: center; font-weight: bold;  font-stretch: normal;  font-style: normal;  line-height: normal;  letter-spacing: normal;  color:'+ captionColor + ';position:absolute;top:0%;left:50%;width:444px;transform:translate( -50%, 124px);');
    popup.appendChild(caption);

    var textColor = isLightThemeEnabled ? '#181a1d' : '#fff';

    var text = document.createElement('span');
    text.setAttribute('class', 'unselectable');
    text.innerHTML = "We are sorry, this feature is not implemented in WEB version.<br/>Please install DESKTOP version (it is really fast)";
    text.setAttribute('style', 'font-family: SegoeUIVariable-Display;font-size: 14px;  text-align: center; font-stretch: normal;  font-style: normal;  line-height: normal;  letter-spacing: normal;  color:'+ textColor + ';position:absolute;top:0%;left:50%;width:444px;transform:translate( -50%, 175px);');
    popup.appendChild(text);

    var button = document.createElement('a');
    button.setAttribute('class', "button");
    button.setAttribute('target', "_blank");
    button.setAttribute('href', 'https://meshinspector.com/download');
    button.innerHTML = "Download";
    button.setAttribute('style', 'text-decoration:none;font-family:SegoeUIVariable-Display;border-radius:4px;cursor:pointer;position:absolute;top:0%;left:50%;width:130px;height:28px;transform:translate( -50%, 243px);font-size:14px;font-stretch:normal;font-style:normal;line-height:normal;letter-spacing:normal;color:#fff;display:inline-flex;align-items:center;justify-content:center;');

    popup.appendChild(button);

    overlay.appendChild( popup );
    document.body.appendChild(overlay);
}
