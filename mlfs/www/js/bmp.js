draw = (t, h, w) => {
    var c = document.createElement('canvas');
    c.id = "CursorLayer";
    c.width = w;
    c.height = h;
    // c.style.zIndex = 8;
    // c.style.position = "absolute";
    // c.style.border = "1px solid";
    // c.fill

    var ctx = c.getContext("2d");

    ctx.beginPath();
    ctx.rect(0, 0, w, h);
    ctx.fillStyle = "red";
    ctx.fill();
    return c;
};

main = () => {
    c = draw(0, 128, 1024);
    document.body.appendChild(c);
};

window.onload = () => {
    // console.log(document.body);
    main()
}
