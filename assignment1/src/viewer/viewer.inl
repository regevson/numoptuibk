template<class InputIt, class UnaryFunction>
void Viewer::drawPoints(InputIt first, InputIt last, UnaryFunction f)
{
    float renderscale = renderScale();
    glPointSize(renderscale * 2.0f * mRender.pointRadius);
    glm::vec2 coord;
    glm::vec4 color;

    glBegin(GL_POINTS);
    for (; first != last; ++first)
    {
        if(!f(*first, coord, color)) continue;

        glColor4f(color.r, color.g, color.b, color.a);
        glVertex2f(renderscale * coord.x, renderscale * coord.y);
    }
    glEnd();
}

template<class InputIt, class UnaryFunction>
void Viewer::drawLines(InputIt first, InputIt last, UnaryFunction f)
{    
    float renderscale = renderScale();
    glLineWidth(renderscale * mRender.lineWidth);
    glm::vec2 start;
    glm::vec2 end;
    glm::vec4 color;

    glBegin(GL_LINES);
    for (; first != last; ++first)
    {
        if(!f(*first, start, end, color)) continue;

        glColor4f(color.r, color.g, color.b, color.a);
        glVertex2f(renderscale * start.x, renderscale * start.y);
        glVertex2f(renderscale * end.x, renderscale * end.y);
    }
    glEnd();
}


template<class InputIt, class UnaryFunction>
void Viewer::drawTriangles(InputIt first, InputIt last, UnaryFunction f)
{
    float renderscale = renderScale();
    glLineWidth(renderscale * mRender.lineWidth);
    glm::vec2 p0;
    glm::vec2 p1;
    glm::vec2 p2;
    glm::vec4 color;

    if(mRender.mWireframe) glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );

    glBegin(GL_TRIANGLES);
    for (; first != last; ++first)
    {
        if(!f(*first, p0, p1, p2, color)) continue;

        glColor4f(color.r, color.g, color.b, color.a);

        glVertex2f(renderscale * p0.x, renderscale * p0.y);
        glVertex2f(renderscale * p1.x, renderscale * p1.y);
        glVertex2f(renderscale * p2.x, renderscale * p2.y);
    }
    glEnd();

    if(mRender.mWireframe) glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
}

template<class InputIt, class UnaryFunction>
void Viewer::drawOutline(InputIt first, InputIt last, UnaryFunction f)
{
    float renderscale = renderScale();
    glLineWidth(renderscale * mRender.lineWidth);
    std::vector<glm::vec2> vertices;
    glm::vec4 color;

    for (; first != last; ++first)
    {
        vertices.clear();
        if(!f(*first, vertices, color)) continue;

        glBegin(GL_LINE_LOOP);

        glColor4f(color.r, color.g, color.b, color.a);

        for(auto& v : vertices)
        {
            glVertex2f(renderscale * v.x, renderscale * v.y);
        }

        glEnd();
    }
}
