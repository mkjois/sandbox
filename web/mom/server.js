var Static = require("node-static"),
    Http = require("http"),
    Files = new Static.Server(),
    IpAddress = process.env.OPENSHIFT_NODEJS_IP || process.env.OPENSHIFT_INTERNAL_IP,
    Port = 8080;

function onRequest (request, response) {

    function debugServe (error, data) {
        if (error) {
            console.error("Error serving %s - %s ...", request.url, error.message);
        } else {
            console.log("Request from %s ...", request.url);
        }
    }

    if (request.url === "/favicon.ico") {
        response.writeHead(200, {'Content-Type': 'image/x-icon'} );
        response.end();
        console.log('favicon request handled ...');
        return;
    }

    Files.serve(request, response, debugServe);
}

var Server = Http.createServer(onRequest);
Server.listen(Port, IpAddress);
console.log("MJ's custom node.js HTTP server running at http://%s:%d/ ...", IpAddress, Port);
