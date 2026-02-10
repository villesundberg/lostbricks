import http.server, ssl, sys

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8443
handler = http.server.SimpleHTTPRequestHandler
httpd = http.server.HTTPServer(('0.0.0.0', port), handler)
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain('cert.pem', 'key.pem')
httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
print(f'Serving HTTPS on port {port}')
httpd.serve_forever()
