var themeChangedCallback = function () {
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        // dark mode
        Module.ccall('emsChangeColorTheme', 'number', ['number'], [0]);
    }
    else {
        // light mode
        Module.ccall('emsChangeColorTheme', 'number', ['number'], [1]);
    }
}

var setupColorTheme = function () {
    themeChangedCallback();
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
        themeChangedCallback();
    });
}

var getColorTheme = function() {
    return Module.ccall( 'emsGetColorTheme' );
}
