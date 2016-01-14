
typedef struct tag_result{
    int xpos, ypos;
	float SAD;
}result;

    // OpenCL Kernel Function for element by element
 kernel void pattern(global uchar* imageData,global uchar* templateData, global float*aux, int t_cols, int t_rows, int w, int h, global result* res)
{
        *aux=(float)105;
        // get index into global data array
        int iGIDX = get_global_id(0);
		int iGIDY = get_global_id(1);
		int iGID = (iGIDY * w + iGIDX);

		int x = iGIDX;
		int y = iGIDY;

        res->xpos=0;
        res->ypos=0;

        float SAD = 0;

	// loop through the template image

		for ( int y1 = 0; y1 < t_rows; y1++ )
            for ( int x1 = 0; x1 < t_cols; x1++ )
			{

				int p_SearchIMG = imageData[(y+y1) * w + (x+x1)];
                int p_TemplateIMG = templateData[y1 *  t_cols + x1];

                SAD += abs( p_SearchIMG - p_TemplateIMG );
            }

        // save the best found position
        if ( (*res).SAD > SAD )
		{
            (*res).SAD = SAD;
            // give me min SAD
			(*res).xpos = x;
            (*res).ypos = y;

        }

        (*res).SAD= 3450;


    }
